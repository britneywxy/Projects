import java.util.*;

public interface SeamFinder {

    // Calculates and returns a minimum-energy horizontal seam in the current image. The returned
    // array will have the same length as the width of the image. A value of v at index i of the
    // output indicates that pixel (i, v) is in the seam.
    List<Integer> findSeam(Picture picture, EnergyFunction f);
}
