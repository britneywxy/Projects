@FunctionalInterface
public interface EnergyFunction {

    // Returns the energy of pixel (x, y) in the given picture. Throws IndexOutOfBoundsException if
    // (x, y) is not inside of the given image, or if the image has fewer than 3 x or y pixels.
    double apply(Picture picture, int x, int y);
}
