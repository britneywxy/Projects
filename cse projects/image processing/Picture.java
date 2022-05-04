import java.awt.image.*;
import java.io.*;
import java.util.*;
import javax.imageio.*;

public class Picture {
    protected final BufferedImage image;

    protected Picture() {
        // For subclassing behavior
        image = null;
    }

    public Picture(BufferedImage image) {
        this.image = image;
    }

    public Picture(int width, int height) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Dimensions must be positive");
        }
        image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    }

    public Picture(Picture other) {
        this(other.width(), other.height());
        for (int i = 0; i < other.width(); i += 1) {
            for (int j = 0; j < other.height(); j += 1) {
                this.set(i, j, other.get(i, j));
            }
        }
    }

    public Picture(File file) throws IOException {
        image = ImageIO.read(file);
    }

    // https://codereview.stackexchange.com/a/244139
    public static Picture random(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        long bytesPerPixel = 4L;
        for (int y = 0; y < height; y += 1) {
            int[] row = new SplittableRandom().ints(bytesPerPixel * width, 0, 256).toArray();
            image.getRaster().setPixels(0, y, width, 1, row);
        }
        return new Picture(image);
    }

    public int get(int x, int y) {
        return image.getRGB(x, y);
    }

    public void set(int x, int y, int rgb) {
        image.setRGB(x, y, rgb);
    }

    public int width() {
        return image.getWidth();
    }

    public int height() {
        return image.getHeight();
    }

    public void save(File file) throws IOException {
        String extension = file.getName().substring(file.getName().lastIndexOf('.') + 1);
        if ("jpg".equalsIgnoreCase(extension) || "png".equalsIgnoreCase(extension)) {
            ImageIO.write(image, extension, file);
        } else {
            throw new IllegalArgumentException("File must end in .jpg or .png");
        }
    }
}
