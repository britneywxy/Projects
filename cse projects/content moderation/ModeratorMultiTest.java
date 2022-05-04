import java.io.*;
import java.util.*;

public class ModeratorMultiTest {

    // Add all comments + toxicities to the given PQ.
    private static void addAllComments(Map<Double, String> toxic, ExtrinsicMinPQ<String> pq) {
        for (double toxicity : toxic.keySet()) {
            String comment = toxic.get(toxicity);
            pq.add(comment, -toxicity);
        }
    }

    // Remove all elements from both reference and test PQ, make sure order matched.
    private static boolean sameRemoveOrder(ExtrinsicMinPQ<String> refPQ, ExtrinsicMinPQ<String> testPQ) {
        int numItems = refPQ.size();
        List<String> refOutput = new ArrayList<>();
        List<String> testOutput = new ArrayList<>();
        for (int i = 0; i < numItems; i++) {
            refOutput.add(refPQ.removeMin());
            testOutput.add(testPQ.removeMin());
        }
        return refOutput.equals(testOutput);
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        // We go toxicity -> comment because all toxicity values used for testing need to be unique.
        Map<Double, String> toxic = new LinkedHashMap<>();

        Scanner scanner = new Scanner(new File("toxic.tsv"));
        scanner.nextLine(); // Skip header
        while (scanner.hasNextLine()) {
            Scanner line = new Scanner(scanner.nextLine()).useDelimiter("\t");
            double toxicity = line.nextDouble();
            String comment = line.next();
            toxic.put(toxicity, comment);
        }

        System.out.println(toxic.size() + " comments loaded");

        // Testing implementations.
        Map<String, ExtrinsicMinPQ<String>> implementations = Map.of(
            "UnsortedArrayMinPQ", new UnsortedArrayMinPQ<String>(),
            "HeapMinPQ", new HeapMinPQ<String>(),
            "OptimizedHeapMinPQ", new OptimizedHeapMinPQ<String>()
        );

        // Check each implementation against the reference matches.
        for (String name : implementations.keySet()) {
            long startTime = System.nanoTime();
            System.out.println();

            ExtrinsicMinPQ<String> referencePQ = new DoubleMapMinPQ<String>();
            ExtrinsicMinPQ<String> testPQ = implementations.get(name);

            // Add same elements to both reference and student-written PQ.
            addAllComments(toxic, referencePQ);
            addAllComments(toxic, testPQ);

            // Test 1: Basic test that `contains` is working.
            boolean containsPassed = true;
            for (double toxicity : toxic.keySet()) {
                String comment = toxic.get(toxicity);

                if (!testPQ.contains(comment)) {
                    System.out.println(name + " contains() FAIL [after insertion]!");
                    containsPassed = false;
                    break;
                }
            }

            if (containsPassed)
                System.out.println(name + " contains() PASS [after insertion]!");

            // Test 2: See if size() is working.
            if (referencePQ.size() == testPQ.size())
                System.out.println(name + " size() PASS [after insertion]!");
            else
                System.out.println(name + " size() FAIL [after insertion]!");

            // Test 3: See if peek() is working.
            if (referencePQ.peekMin().equals(testPQ.peekMin()))
                System.out.println(name + " peek() PASS!");
            else
                System.out.println(name + " peek() FAIL!");

            // Test 4: Remove all comments from both the reference and test PQ, make sure
            // the order matches.
            if (sameRemoveOrder(referencePQ, testPQ))
                System.out.println(name + " removeMin() PASS!");
            else
                System.out.println(name + " removeMin() FAIL!");

            // Test 5: Make sure contains works after removing everything.
            containsPassed = true;
            for (double toxicity : toxic.keySet()) {
                String comment = toxic.get(toxicity);

                // This should not happen on an emptied-out PQ.
                if (testPQ.contains(comment)) {
                    System.out.println(name + " contains() FAIL [after removeMin]!");
                    containsPassed = false;
                    break;
                }
            }

            if (containsPassed)
                System.out.println(name + " contains() PASS [after removeMin]!");

            // Test 6: Make sure size works after removing everything.
            if (referencePQ.size() == testPQ.size())
                System.out.println(name + " size() PASS [after removeMin]!");
            else
                System.out.println(name + " size() FAIL [after removeMin]!");

            // Test 7: Re-populate, change priorities around, test new remove order.
            addAllComments(toxic, referencePQ);
            addAllComments(toxic, testPQ);

            // Mix up all the priority values.
            List<String> shuffledComments = new ArrayList<>();
            List<Double> shuffledPriorities = new ArrayList<>();
            for (double toxicity : toxic.keySet()) {
                String comment = toxic.get(toxicity);
                shuffledComments.add(comment);
                shuffledPriorities.add(toxicity);
            }
            Collections.shuffle(shuffledPriorities);

            // Change priority.
            for (int i = 0; i < shuffledComments.size(); i++) {
                referencePQ.changePriority(shuffledComments.get(i), shuffledPriorities.get(i));
                testPQ.changePriority(shuffledComments.get(i), shuffledPriorities.get(i));
            }

            // Make sure order matches after `changePriority`.
            if (sameRemoveOrder(referencePQ, testPQ))
                System.out.println(name + " changePriority() PASS!");
            else
                System.out.println(name + " changePriority() FAIL!");

            System.out.println();
            long duration = System.nanoTime() - startTime;
            System.out.println(name + " " + duration/10000000);
        }
    }
}

