import graphs.*;

import java.util.*;

public interface Node {

    // Returns all of the adjacent and diagnoally-adjacent neighbors of this node.
    List<Edge<Node>> neighbors(Picture picture, EnergyFunction f);
}
