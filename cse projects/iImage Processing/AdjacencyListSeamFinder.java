import graphs.*;

import java.util.*;

public class AdjacencyListSeamFinder implements SeamFinder {
    private final ShortestPathSolver.Constructor<Node> sps;

    public AdjacencyListSeamFinder(ShortestPathSolver.Constructor<Node> sps) {
        this.sps = sps;
    }

    public List<Integer> findSeam(Picture picture, EnergyFunction f) {
        PixelGraph graph = new PixelGraph(picture, f);
        List<Node> seam = sps.run(graph, graph.source).solution(graph.sink);
        seam = seam.subList(1, seam.size() - 1);
        List<Integer> result = new ArrayList<>(seam.size());
        for (Node pixel : seam) {
            result.add(((PixelGraph.Pixel) pixel).y);
        }
        return result;
    }

    private class PixelGraph implements Graph<Node> {
        public final Picture picture;
        public final EnergyFunction f;
        public final Pixel[][] pixels;

        public PixelGraph(Picture picture, EnergyFunction f) {
            this.picture = picture;
            this.f = f;
            this.pixels = new Pixel[picture.width()][picture.height()];
            
            for (int y = 0; y < picture.height(); y += 1) {
                Pixel from = new Pixel(picture.width() - 1, y);
                from.addEdge(new Edge<>(from, sink, 0));
                pixels[picture.width() - 1][y] = from;
            }
            for (int x = picture.width() - 2; x >= 0; x -= 1) {
                for (int y = 0; y < picture.height(); y += 1) {
                    Pixel from = new Pixel(x, y);
                    for (int z = y - 1; z <= y + 1; z += 1) {
                        if (0 <= z && z < picture.height()) {
                            Pixel to = pixels[x + 1][z];
                            from.addEdge(new Edge<>(from, to, f.apply(picture, x + 1, z)));
                        }
                    }
                    pixels[x][y] = from;
                }
            }
        }

        public List<Edge<Node>> neighbors(Node node) {
            return node.neighbors(picture, f);
        }

        public final Node source = new Node() {
            public List<Edge<Node>> neighbors(Picture picture, EnergyFunction f) {
                List<Edge<Node>> result = new ArrayList<>(picture.height());
                for (int j = 0; j < picture.height(); j += 1) {
                    Pixel to = pixels[0][j];
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
            public List<Edge<Node>> neighbors;

            public Pixel(int x, int y) {
                this.x = x;
                this.y = y;
                this.neighbors = new ArrayList<>(3);
            }

            public void addEdge(Edge<Node> edge) {
                neighbors.add(edge);
            }

            public List<Edge<Node>> neighbors(Picture picture, EnergyFunction f) {
                return neighbors;
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
