import java.util.*;

class PriorityNode<T> {
    private final T item;
    private double priority;

    PriorityNode(T item, double priority) {
        this.item = item;
        this.priority = priority;
    }

    T item() {
        return item;
    }

    double priority() {
        return priority;
    }

    void setPriority(double priority) {
        this.priority = priority;
    }

    public boolean equals(Object o) {
        if (this == o) {
            return true;
        } else if (!(o instanceof PriorityNode)) {
            return false;
        }
        PriorityNode other = (PriorityNode) o;
        return Objects.equals(this.item, other.item);
    }

    public int hashCode() {
        return item.hashCode();
    }

    public String toString() {
        return item + " (" + priority + ')';
    }
}
