import graphs.Graph;

import java.util.*;

public class DynamicProgrammingSeamFinder implements SeamFinder {

    public List<Integer> findSeam(Picture picture, EnergyFunction f) {
        // TODO: Your code here!
        double[][] energyTable = new double[picture.width()][picture.height()];
        for(int x = 0; x < picture.width(); x++){
            for(int y = 0; y < picture.height(); y++){
                //apply the energy
                double currEnergy = f.apply(picture, x, y);
                //Left edge
                if(x == 0){
                    energyTable[x][y] = currEnergy;
                } else {  //every other edges
                    if(y == 0){ //upper edge
                        double left = energyTable[x-1][y];
                        double leftDown = energyTable[x-1][y+1];
                        min = Math.min(left,leftDown);
                    } else if (y == picture.height() - 1){  //lower edge
                        double left = energyTable[x-1][y];
                        double leftUp = energyTable[x-1][y-1];
                        min = Math.min(left,leftUp);
                    } else {  //other edges
                        double left = energyTable[x-1][y];
                        double leftUp = energyTable[x-1][y-1];
                        double leftDown = energyTable[x-1][y+1];
                        double tempMin = Math.min(left,leftUp);
                        min = Math.min(tempMin,leftDown);
                    } 
                    energyTable[x][y] = min + currEnergy;
                }
            }
        }
            
        //compute the shortest path
        List<Integer> result = new ArrayList<>();
        //find the smallest node in the last edge
        double minEdge = Double.POSITIVE_INFINITY;
        int smallY = 0;
        for(int y = 0; y < picture.height(); y++){
            double curr = energyTable[picture.width()-1][y];
            if(curr < minEdge){
                minEdge = curr;
                smallY = y;
            }
        }
        result.add(smallY);

        // find the shortest path: start from right to left
        // adding the y-coordinate of each minimum-cost predecessor to a list.
        int prevY = smallY;
        for(int x = picture.width()-2; x >= 0; x--){
            double smallEdge = Double.POSITIVE_INFINITY;
            int minY = 0;
            // value Y only focused on the three neightbors: left,leftUp,leftDown
            for(int y = prevY - 1; y <= prevY + 1; y++){
                if(y >= 0 && y<picture.height()){
                    if(energyTable[x][y] < smallEdge){
                        smallEdge = energyTable[x][y];
                        minY = y;
                    } 
                }
            }
            prevY = minY;
            result.add(minY);
        }

        //reverse the list
        Collections.reverse(result);
        return result;
    }
}
