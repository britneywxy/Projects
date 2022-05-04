import graphs.*;

import java.util.*;

public class GenerativeSeamFinder implements SeamFinder {
    private final ShortestPathSolver.Constructor<Node> sps;

    public GenerativeSeamFinder(ShortestPathSolver.Constructor<Node> sps) {
        this.sps = sps;
    }

    public List<Integer> findSeam(Picture picture, EnergyFunction f) {
        // TODO: Your code here!
        PixelGraph graph = new PixelGraph(picture, f);
        List<Node> seam = sps.run(graph, graph.source).solution(graph.sink);
        seam = seam.subList(1, seam.size() - 1);  //not include source and sink
        List<Integer> result = new ArrayList<>(seam.size());
        for (Node pixel : seam) {
            result.add(((PixelGraph.Pixel) pixel).y);
        }
        return result;
    }

    // TODO: Your PixelGraph inner class here!
    private class PixelGraph implements Graph<Node> {
        public final Picture picture;
        public final EnergyFunction f;

        public PixelGraph(Picture picture, EnergyFunction f) {
            this.picture = picture;
            this.f = f;
        }

        public List<Edge<Node>> neighbors(Node node) {
            return node.neighbors(picture, f);
        }

        public final Node source = new Node() {
            public List<Edge<Node>> neighbors(Picture picture, EnergyFunction f) {
                List<Edge<Node>> result = new ArrayList<>(picture.height());
                for (int j = 0; j < picture.height(); j += 1) {
                    Pixel to = new Pixel(0,j);
                    result.add(new Edge<>(this, to, f.apply(picture, 0, j)));
                }
                return result;
            }
        };

        public final Node sink = new Node() {
            public List<Edge<Node>> neighbors(Picture picture, EnergyFunction f) {
                return List.of(); // Sink has no neighbors
            }
        };

        public class Pixel implements Node {
            public final int x;
            public final int y;

            public Pixel(int x, int y) {
                this.x = x;
                this.y = y;
            }

            public List<Edge<Node>> neighbors(Picture picture, EnergyFunction f) {
                List<Edge<Node>> edges = new ArrayList<>();
                int neighborX = x + 1;
                if (neighborX > picture.width() - 1){ //last node
                    edges.add(new Edge<Node>(this, sink, 0));
                } else {
                    Pixel from = new Pixel(x, y);
                    for(int neighborY = y-1; neighborY <= y+1; neighborY++){
                        if(neighborY >= 0 && neighborY < picture.height()) {
                            Pixel to = new Pixel(x+1,neighborY);
                            edges.add(new Edge<Node>(from,to,f.apply(picture, x+1, neighborY)));                                                     
                        }
                    }
                }
                return edges;
            }

            public String toString() {
                return "(" + x + ", " + y + ")";
            }

            public boolean equals(Object o) {
                if (this == o) {
                    return true;
                } else if (!(o instanceof Pixel)) {
                    return false;
                }
                Pixel other = (Pixel) o;
                return this.x == other.x && this.y == other.y;
            }

            public int hashCode() {
                return Objects.hash(x, y);
            }
        }
    }

}
