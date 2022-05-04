import java.util.*;

public class OptimizedHeapMinPQ<T> implements ExtrinsicMinPQ<T> {
    private List<PriorityNode<T>> items;
    private Map<T, Integer> itemToIndex;
    private int size;

    public OptimizedHeapMinPQ() {
        items = new ArrayList<>();
        itemToIndex = new HashMap<>();
    }

    public void add(T item, double priority) {
        if (item == null) {
            throw new IllegalArgumentException("Item is null");
        }
        if (contains(item)) {
            throw new IllegalArgumentException("Already contains " + item);
        }

        int curPlace = size();

        items.add(new PriorityNode<T>(item, priority)); //add to list
        itemToIndex.put(item, curPlace); //add to map

        while (items.get(curPlace).priority() < items.get(parent(curPlace)).priority()) {
            swap(curPlace, parent(curPlace));
            curPlace = parent(curPlace);
        }
        size++;
    }

    public boolean contains(T item) {
        return itemToIndex.containsKey(item);
    }

    public T peekMin() {
        if (isEmpty()) {
            throw new NoSuchElementException("PQ is empty");
        }
        return items.get(0).item();
    }

    public T removeMin() {
        T minVal = items.get(0).item();
        PriorityNode<T> lastNode = items.get(size()-1);

        //remove in list
        items.set(0, lastNode);
        items.remove(size()-1);

        //remove in map
        itemToIndex.put(lastNode.item(), 0); 
        itemToIndex.remove(minVal);
        heapify(0);
        return minVal;
    }

    public void changePriority(T item, double priority) {
        if (!contains(item)) {
            throw new NoSuchElementException("PQ does not contain " + item);
        }

        int index = itemToIndex.get(item);
        double originalPriority = items.get(index).priority();
        items.set(index, new PriorityNode<T>(item, priority));
        if(priority == originalPriority) return;
        if(priority > originalPriority){
            heapify(index);
        }else if(priority < originalPriority){
            while(items.get(index).priority() < items.get(parent(index)).priority()){
                swap(index, parent(index));
                index = parent(index);
            }
        }
    }

    public int size() {
        size = items.size();
        return size;
    }

    private void swap(int index1, int index2) {
        T item1 = items.get(index1).item();
        T item2 = items.get(index2).item();

        // Swap item in list
        PriorityNode<T> tmp = items.get(index1);
        items.set(index1, items.get(index2));
        items.set(index2, tmp);

        // Swap index in map
        itemToIndex.put(item1, index2);
        itemToIndex.put(item2, index1);
    }

    private int parent(int index) {
        if (index == 0) {
            return 0;
        }
        return (index - 1) / 2;
    }

    private int leftChild(int index) {
        return (index * 2) + 1;
    }

    private int rightChild(int index) {
        return (index * 2) + 2;
    }

    private boolean hasOnlyOneNode(int curPlace) {
        size = size();
        if (rightChild(curPlace)>=size || leftChild(curPlace)>=size) {
            return true;
        }
        return false;
    }

    private void heapify(int curPlace) {
        if (!hasOnlyOneNode(curPlace)) {
            if (items.get(curPlace).priority() > items.get(leftChild(curPlace)).priority()
                    || items.get(curPlace).priority() > items.get(rightChild(curPlace)).priority()) {
                if (items.get(leftChild(curPlace)).priority() < items.get(rightChild(curPlace)).priority()) {
                    swap(curPlace, leftChild(curPlace));
                    heapify(leftChild(curPlace));
                } else{
                    swap(curPlace, rightChild(curPlace));
                    heapify(rightChild(curPlace));
                }
            }
        }
        if (hasOnlyLeftChild(curPlace)){
            if (items.get(leftChild(curPlace)).priority() < items.get(curPlace).priority()) {
                swap(curPlace, leftChild(curPlace));
            }
        }
    }
    
    private boolean hasOnlyLeftChild(int curPlace){
        size = size();
        if(leftChild(curPlace)<size && rightChild(curPlace)>=size){
            return true;
        }
        return false;
    }
}
