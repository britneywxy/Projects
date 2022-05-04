public class DualGradientEnergyFunction implements EnergyFunction {

    public double apply(Picture picture, int x, int y) {
        if (x < 0 || y < 0 || x >= picture.width() || y >= picture.height()) {
            throw new IndexOutOfBoundsException("Invalid indices for given picture");
        }
        return Math.sqrt(xDerivative(picture, x, y) + yDerivative(picture, x, y));
    }

    private static double xDerivative(Picture picture, int x, int y) {
        if (x == 0) {
            return forwardDiff(picture.get(x, y), picture.get(x + 1, y), picture.get(x + 2, y));
        } else if (x == picture.width() - 1) {
            return forwardDiff(picture.get(x, y), picture.get(x - 1, y), picture.get(x - 2, y));
        } else {
            return centralDiff(picture.get(x - 1, y), picture.get(x + 1, y));
        }
    }

    private static double yDerivative(Picture picture, int x, int y) {
        if (y == 0) {
            return forwardDiff(picture.get(x, y), picture.get(x, y + 1), picture.get(x, y + 2));
        } else if (y == picture.height() - 1) {
            return forwardDiff(picture.get(x, y), picture.get(x, y - 1), picture.get(x, y - 2));
        } else {
            return centralDiff(picture.get(x, y - 1), picture.get(x, y  + 1));
        }
    }

    private static double forwardDiff(int rgb1, int rgb2, int rgb3) {
        return Math.pow(4 * red(rgb2)   - red(rgb3)   - 3 * red(rgb1),   2)
             + Math.pow(4 * green(rgb2) - green(rgb3) - 3 * green(rgb1), 2)
             + Math.pow(4 * blue(rgb2)  - blue(rgb3)  - 3 * blue(rgb1),  2);
    }

    private static double centralDiff(int rgb1, int rgb2) {
        return Math.pow(red(rgb1)   - red(rgb2),   2)
             + Math.pow(green(rgb1) - green(rgb2), 2)
             + Math.pow(blue(rgb1)  - blue(rgb2),  2);
    }

    private static int red(int rgb) {
        return (rgb >> 16) & 0xFF;
    }

    private static int green(int rgb) {
        return (rgb >> 8) & 0xFF;
    }

    private static int blue(int rgb) {
        return (rgb >> 0) & 0xFF;
    }
}
