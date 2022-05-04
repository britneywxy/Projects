import java.util.*;

public class UnsortedArrayMinPQ<T> implements ExtrinsicMinPQ<T> {
    private List<PriorityNode<T>> items;

    public UnsortedArrayMinPQ() {
        items = new ArrayList<>();
    }

    public void add(T item, double priority) {
        if (contains(item)) {
            throw new IllegalArgumentException("Already contains " + item);
        }
        //initialize node
        PriorityNode<T> newNode = new PriorityNode<T>(item, priority);

        //find insert position
        int insertPosition = 0;
        boolean findGreater = false;
        for(int i =0; i < items.size(); i++){
            PriorityNode<T> curNode = items.get(i);
            if(curNode.priority() > priority){
                insertPosition = i;
                findGreater = true;
                break;
            }
        }

        //Add to the ArrayList
        if(!findGreater){
            items.add(newNode);
        } else {
            items.add(insertPosition, newNode);
        }
    }

    public boolean contains(T item) {
        for (PriorityNode<T> Node : items) {
            if (Node.item().equals(item)) {
                return true;
            }
        }
        return false;
    }

    public T peekMin() {
        if (isEmpty()) {
            throw new NoSuchElementException("PQ is empty");
        }
        return items.get(0).item();
        
    }

    public T removeMin() {
        if (isEmpty()) {
            throw new NoSuchElementException("PQ is empty");
        }
        T firstNode = peekMin();
        items.remove(0);
        return firstNode;
    }

    public void changePriority(T item, double priority) {
        if (!contains(item)) {
            throw new NoSuchElementException("PQ does not contain " + item);
        }
        //find and remove element
        int index = 0;
        for(int i = 0; i < items.size(); i++){
            PriorityNode<T> curNode = items. get(i);
            if(curNode.item() == item){
                index = i;
                break;
            }
        }
        items.remove(index);

        //add element
        add(item,priority);
        
    }

    public int size() {
        return items.size();
    }
}
