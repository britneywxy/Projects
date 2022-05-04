import java.io.*;
import java.util.*;

public class Moderator {
    public static final boolean SAFE_FOR_WORK = true;

    public static void main(String[] args) throws IOException, InterruptedException {
        Scanner scanner = new Scanner(new File("toxic.tsv"));
        scanner.nextLine(); // Skip header

        // Simulate a moderation workflow where comments are streaming-in.
        //ExtrinsicMinPQ<String> pq = new UnsortedArrayMinPQ<>();
        ExtrinsicMinPQ<String> pq = new HeapMinPQ<>();
        Random random = new Random();
        addComments(pq, scanner, random.nextInt(100));
        Scanner stdin = new Scanner(System.in);
        while (!pq.isEmpty()) {
            System.out.println();
            if (SAFE_FOR_WORK) {
                System.out.println(pq.removeMin().replaceAll("\\B[a-zA-Z]", "*"));
            } else {
                System.out.println(pq.removeMin());
            }
            System.out.print("[Y]es/[N]o: ");
            String response = null;
            while (response == null && stdin.hasNextLine()) {
                response = stdin.nextLine();
                switch (response.strip().toLowerCase()) {
                    case "y": case "yes":
                    case "n": case "no":
                        // In a real system, write the response to the database.
                        break;
                    default:
                        response = null;
                        System.out.print("[Y]es/[N]o: ");
                        break;
                }
            }
            if (random.nextBoolean()) {
                addComments(pq, scanner, random.nextInt(4));
            }
        }
    }

    // Add up to n comments from the scanner to the pq with the weight negated.
    private static void addComments(ExtrinsicMinPQ<String> pq, Scanner scanner, int n) {
        int i = 0;
        for (; i < n && scanner.hasNextLine(); i += 1) {
            Scanner line = new Scanner(scanner.nextLine()).useDelimiter("\t");
            double toxicity = line.nextDouble();
            String comment = line.next();
            // Prioritize most toxic content first by negating the weight.
            pq.add(comment, -toxicity);
        }
        System.out.println(i + " comments added to pq");
    }
}
