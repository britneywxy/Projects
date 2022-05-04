import java.util.*;

public class HeapMinPQ<T> implements ExtrinsicMinPQ<T> {
    private final PriorityQueue<PriorityNode<T>> pq;

    public HeapMinPQ() {
        pq = new PriorityQueue<>(Comparator.comparingDouble(PriorityNode::priority));
    }

    public void add(T item, double priority) {
        if(item == null){
            throw new IllegalArgumentException("Item is null");
        }
        if (contains(item)) {
            throw new IllegalArgumentException("Already contains " + item);
        }
        pq.add(new PriorityNode<T>(item, priority));
    }

    public boolean contains(T item) {
        for(PriorityNode<T> curNode: pq){
            if(curNode.item().equals(item)){
                return true;
            }
        }
        return false;
    }

    public T peekMin() {
        if (isEmpty()) {
            throw new NoSuchElementException("PQ is empty.");
        }
        return pq.peek().item();
    }

    public T removeMin() {
        if (isEmpty()) {
            throw new NoSuchElementException("PQ is empty.");
        }
        return pq.poll().item();
    }

    public void changePriority(T item, double priority) {
        if(!contains(item)){
            throw new NoSuchElementException("PQ does not contain " + item);
        }
        //make a copy and clear the original pq
        PriorityQueue<PriorityNode<T>> pqCopy = new PriorityQueue<PriorityNode<T>>(pq);
        pq.clear();
        //put in elements to the original pq
        while(!pqCopy.isEmpty()){
            PriorityNode<T> curNode = pqCopy.poll();
            if(curNode.item().equals(item)){
                PriorityNode<T> newNode = new PriorityNode<T>(item, priority);
                pq.add(newNode);
            } else {
                pq.add(curNode);
            }
        }
    }

    public int size() {
        return pq.size();
    }
}
