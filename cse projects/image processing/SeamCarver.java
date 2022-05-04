import graphs.*;

import java.io.*;
import java.util.*;

// Client class implementing seam carving, which combines an energy function and a seam finding
// algorithm to carve seams in a picture.
public class SeamCarver {
    private Picture picture;
    private final EnergyFunction f;
    private final SeamFinder seamFinder;

    public static void main(String[] args) throws IOException {
        EnergyFunction f = new DualGradientEnergyFunction();
        SeamFinder seamFinder = new AdjacencyListSeamFinder(DijkstraSolver::new);
        SeamCarver seamCarver = new SeamCarver(new File("data/HJoceanSmall.png"), f, seamFinder);
        System.out.println(seamCarver.removeVertical());
        seamCarver.picture().save(new File("result.png"));
    }

    public SeamCarver(File file, EnergyFunction f, SeamFinder seamFinder) throws IOException {
        if (file == null || f == null || seamFinder == null) {
            throw new IllegalArgumentException("Arguments cannot be null");
        }
        this.picture = new Picture(file);
        this.f = f;
        this.seamFinder = seamFinder;
    }

    public Picture picture() {
        return new Picture(picture);
    }

    public List<Integer> removeHorizontal() {
        List<Integer> seam = seamFinder.findSeam(picture, f);
        validate(picture, seam);
        Picture result = new Picture(picture.width(), picture.height() - 1);
        for (int x = 0; x < picture.width(); x += 1) {
            for (int y = 0; y < seam.get(x); y += 1) {
                result.set(x, y, picture.get(x, y));
            }
            for (int y = seam.get(x); y < picture.height() - 1; y += 1) {
                result.set(x, y, picture.get(x, y + 1));
            }
        }
        picture = result;
        return seam;
    }

    public List<Integer> removeVertical() {
        Picture transposed = new Picture() {
            public int get(int x, int y) {
                return picture.image.getRGB(y, x);
            }

            public void set(int x, int y, int rgb) {
                throw new UnsupportedOperationException("Transposed picture is immutable");
            }

            public int width() {
                return picture.image.getHeight();
            }

            public int height() {
                return picture.image.getWidth();
            }

            public void save(File file) {
                throw new UnsupportedOperationException("Transposed picture cannot be saved");
            }
        };
        List<Integer> seam = seamFinder.findSeam(transposed, f);
        validate(transposed, seam);
        Picture result = new Picture(picture.width() - 1, picture.height());
        for (int y = 0; y < picture.height(); y += 1) {
            for (int x = 0; x < seam.get(y); x += 1) {
                result.set(x, y, picture.get(x, y));
            }
            for (int x = seam.get(y); x < picture.width() - 1; x += 1) {
                result.set(x, y, picture.get(x + 1, y));
            }
        }
        picture = result;
        return seam;
    }

    private static void validate(Picture picture, List<Integer> seam) {
        if (seam == null) {
            throw new NullPointerException("Seam cannot be null");
        } else if (seam.size() == 1) {
            throw new IllegalArgumentException("Cannot remove seam of length 1");
        } else if (seam.size() != picture.width()) {
            throw new IllegalArgumentException("Seam length does not match image size");
        }
        for (int i = 0; i < seam.size() - 2; i++) {
            if (Math.abs(seam.get(i) - seam.get(i + 1)) > 1) {
                throw new IllegalArgumentException("Seam values too far from adjacent neighbors");
            }
        }
    }
}
