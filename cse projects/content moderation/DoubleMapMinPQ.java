import java.util.*;

public class DoubleMapMinPQ<T> implements ExtrinsicMinPQ<T> {
    private final TreeMap<Double, Set<T>> priorityToItem;
    private final HashMap<T, Double> itemToPriority;

    public DoubleMapMinPQ() {
        priorityToItem = new TreeMap<>();
        itemToPriority = new HashMap<>();
    }

    public void add(T item, double priority) {
        if (contains(item)) {
            throw new IllegalArgumentException("Already contains " + item);
        }
        if (!priorityToItem.containsKey(priority)) {
            priorityToItem.put(priority, new HashSet<>());
        }
        Set<T> itemsWithPriority = priorityToItem.get(priority);
        itemsWithPriority.add(item);
        itemToPriority.put(item, priority);
    }

    public boolean contains(T item) {
        return itemToPriority.containsKey(item);
    }

    public T peekMin() {
        if (isEmpty()) {
            throw new NoSuchElementException("PQ is empty");
        }
        double minPriority = priorityToItem.firstKey();
        Set<T> itemsWithMinPriority = priorityToItem.get(minPriority);
        return firstOf(itemsWithMinPriority);
    }

    public T removeMin() {
        if (isEmpty()) {
            throw new NoSuchElementException("PQ is empty");
        }
        double minPriority = priorityToItem.firstKey();
        Set<T> itemsWithMinPriority = priorityToItem.get(minPriority);
        T item = firstOf(itemsWithMinPriority);
        itemsWithMinPriority.remove(item);
        if (itemsWithMinPriority.isEmpty()) {
            priorityToItem.remove(minPriority);
        }
        itemToPriority.remove(item);
        return item;
    }

    public void changePriority(T item, double priority) {
        if (!contains(item)) {
            throw new NoSuchElementException("PQ does not contain " + item);
        }
        double oldPriority = itemToPriority.get(item);
        if (priority != oldPriority) {
            Set<T> itemsWithOldPriority = priorityToItem.get(oldPriority);
            itemsWithOldPriority.remove(item);
            if (itemsWithOldPriority.isEmpty()) {
                priorityToItem.remove(oldPriority);
            }
            itemToPriority.remove(item);
            add(item, priority);
        }
    }

    public int size() {
        return itemToPriority.size();
    }

    private T firstOf(Set<T> set) {
        return set.iterator().next();
    }
}
