#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_KEYS 3
#define MIN_KEYS 1

/* Node structure */
typedef struct Node {
    bool is_leaf;
    int n;
    int keys[MAX_KEYS];
    int values[MAX_KEYS];
    struct Node *children[MAX_KEYS + 1];
    struct Node *next;
} Node;

Node *root = NULL;


/* Utility */

Node* create_node(bool leaf) {
    Node *node = (Node*)malloc(sizeof(Node));
    node->is_leaf = leaf;
    node->n = 0;
    node->next = NULL;
    for (int i = 0; i < MAX_KEYS; i++) {
        node->keys[i] = 0;
        node->values[i] = 0;
    }
    for (int i = 0; i <= MAX_KEYS; i++)
        node->children[i] = NULL;
    return node;
}

/* Safe integer input */

int safe_get_int(const char *prompt) {
    int val, c;
    while (1) {
        printf("%s", prompt);
        int result = scanf("%d", &val);
        if (result == 1) {
            // Consume any remaining characters on the line
            while ((c = getchar()) != '\n' && c != EOF);
            return val;
        } else {
            // Clear the invalid input buffer
            while ((c = getchar()) != '\n' && c != EOF);
            if (c == EOF) {
                printf("EOF detected. Exiting.\n");
                exit(0);
            }
            printf("Invalid input. Please enter an integer.\n");
        }
    }
}

/* Search */
int search(Node *node, int key) {
    if (!node) return -1;
    int i = 0;
    while (i < node->n && key > node->keys[i]) i++;

    if (node->is_leaf) {
        return (i < node->n && key == node->keys[i]) ? node->values[i] : -1;
    }
    return search(node->children[i], key);
}

/* Insertion Helpers */

void insert_into_leaf(Node *leaf, int key, int value) {
    int i = leaf->n - 1;
    while (i >= 0 && key < leaf->keys[i]) {
        leaf->keys[i + 1] = leaf->keys[i];
        leaf->values[i + 1] = leaf->values[i];
        i--;
    }
    leaf->keys[i + 1] = key;
    leaf->values[i + 1] = value;
    leaf->n++;
}

void insert_into_internal(Node *parent, int key, Node *right_child) {
    int i = parent->n - 1;
    while (i >= 0 && key < parent->keys[i]) {
        parent->keys[i + 1] = parent->keys[i];
        parent->children[i + 2] = parent->children[i + 1];
        i--;
    }
    parent->keys[i + 1] = key;
    parent->children[i + 2] = right_child;
    parent->n++;
}

bool insert_recursive(Node *node, int key, int value, int *promoted_key, Node **new_child) {
    if (node->is_leaf) {
        if (node->n < MAX_KEYS) {
            insert_into_leaf(node, key, value);
            return false;
        }

        int temp_keys[MAX_KEYS + 1], temp_vals[MAX_KEYS + 1];
        int i = 0, j = 0;
        while (i < node->n && node->keys[i] < key) {
            temp_keys[j] = node->keys[i];
            temp_vals[j] = node->values[i];
            i++; j++;
        }
        temp_keys[j] = key; temp_vals[j] = value; j++;
        while (i < node->n) {
            temp_keys[j] = node->keys[i];
            temp_vals[j] = node->values[i];
            i++; j++;
        }

        int mid = (MAX_KEYS + 1) / 2;
        for (i = 0; i < mid; i++) {
            node->keys[i] = temp_keys[i];
            node->values[i] = temp_vals[i];
        }
        node->n = mid;

        Node *new_leaf = create_node(true);
        for (i = mid, j = 0; i < MAX_KEYS + 1; i++, j++) {
            new_leaf->keys[j] = temp_keys[i];
            new_leaf->values[j] = temp_vals[i];
        }
        new_leaf->n = MAX_KEYS + 1 - mid;

        new_leaf->next = node->next;
        node->next = new_leaf;

        *promoted_key = new_leaf->keys[0];
        *new_child = new_leaf;
        return true;
    }

    int i = 0;
    while (i < node->n && key > node->keys[i]) i++;
    Node *child = node->children[i];

    int child_promoted_key;
    Node *new_child_from_child;
    bool split = insert_recursive(child, key, value, &child_promoted_key, &new_child_from_child);

    if (!split) return false;

    if (node->n < MAX_KEYS) {
        insert_into_internal(node, child_promoted_key, new_child_from_child);
        return false;
    }

    int temp_keys[MAX_KEYS + 1];
    Node *temp_children[MAX_KEYS + 2];

    int pos = 0;
    for (int k = 0; k < node->n; k++) {
        temp_keys[pos] = node->keys[k];
        temp_children[pos] = node->children[k];
        pos++;
    }
    temp_children[node->n] = node->children[node->n];

    int insert_pos = 0;
    while (insert_pos < node->n && child_promoted_key > temp_keys[insert_pos])
        insert_pos++;
    for (int k = node->n; k > insert_pos; k--) {
        temp_keys[k] = temp_keys[k - 1];
        temp_children[k + 1] = temp_children[k];
    }
    temp_keys[insert_pos] = child_promoted_key;
    temp_children[insert_pos + 1] = new_child_from_child;

    int mid = MAX_KEYS / 2;

    for (int k = 0; k < mid; k++) {
        node->keys[k] = temp_keys[k];
        node->children[k] = temp_children[k];
    }
    node->children[mid] = temp_children[mid];
    node->n = mid;

    Node *new_internal = create_node(false);
    int j = 0;
    for (int k = mid + 1; k < MAX_KEYS; k++) {
        new_internal->keys[j] = temp_keys[k];
        new_internal->children[j] = temp_children[k];
        j++;
    }
    new_internal->children[j] = temp_children[MAX_KEYS];
    new_internal->n = j;

    *promoted_key = temp_keys[mid];
    *new_child = new_internal;
    return true;
}

void insert(int key, int value) {
    if (root == NULL) {
        root = create_node(true);
        root->keys[0] = key;
        root->values[0] = value;
        root->n = 1;
        return;
    }

    int promoted_key;
    Node *new_child;
    if (insert_recursive(root, key, value, &promoted_key, &new_child)) {
        Node *new_root = create_node(false);
        new_root->keys[0] = promoted_key;
        new_root->children[0] = root;
        new_root->children[1] = new_child;
        new_root->n = 1;
        root = new_root;
    }
}

/* Deletion Helpers */

int find_key_index(Node *node, int key) {
    for (int i = 0; i < node->n; i++) {
        if (node->keys[i] == key) return i;
        if (key < node->keys[i]) return -1;
    }
    return -1;
}

void remove_from_leaf(Node *leaf, int key) {
    int idx = 0;
    while (idx < leaf->n && leaf->keys[idx] != key) idx++;
    if (idx == leaf->n) return;
    for (int i = idx; i < leaf->n - 1; i++) {
        leaf->keys[i] = leaf->keys[i + 1];
        leaf->values[i] = leaf->values[i + 1];
    }
    leaf->n--;
}

void borrow_from_left_leaf(Node *leaf, Node *left_sibling, Node *parent, int parent_idx) {
    int last_key = left_sibling->keys[left_sibling->n - 1];
    int last_val = left_sibling->values[left_sibling->n - 1];

    for (int i = leaf->n; i > 0; i--) {
        leaf->keys[i] = leaf->keys[i - 1];
        leaf->values[i] = leaf->values[i - 1];
    }
    leaf->keys[0] = last_key;
    leaf->values[0] = last_val;
    leaf->n++;
    left_sibling->n--;
    parent->keys[parent_idx] = leaf->keys[0];
}

void borrow_from_right_leaf(Node *leaf, Node *right_sibling, Node *parent, int parent_idx) {
    leaf->keys[leaf->n] = right_sibling->keys[0];
    leaf->values[leaf->n] = right_sibling->values[0];
    leaf->n++;

    for (int i = 0; i < right_sibling->n - 1; i++) {
        right_sibling->keys[i] = right_sibling->keys[i + 1];
        right_sibling->values[i] = right_sibling->values[i + 1];
    }
    right_sibling->n--;
    parent->keys[parent_idx] = right_sibling->keys[0];
}

void merge_leaf_nodes(Node *left, Node *right, Node *parent, int parent_idx) {
    for (int i = 0; i < right->n; i++) {
        left->keys[left->n + i] = right->keys[i];
        left->values[left->n + i] = right->values[i];
    }
    left->n += right->n;
    left->next = right->next;

    for (int i = parent_idx; i < parent->n - 1; i++) {
        parent->keys[i] = parent->keys[i + 1];
        parent->children[i + 1] = parent->children[i + 2];
    }
    parent->n--;
    free(right);
}

void borrow_from_left_internal(Node *internal, Node *left_sibling, Node *parent, int parent_idx) {
    for (int i = internal->n; i > 0; i--) {
        internal->keys[i] = internal->keys[i - 1];
        internal->children[i + 1] = internal->children[i];
    }
    internal->children[1] = internal->children[0];
    internal->keys[0] = parent->keys[parent_idx];
    internal->children[0] = left_sibling->children[left_sibling->n];
    internal->n++;
    parent->keys[parent_idx] = left_sibling->keys[left_sibling->n - 1];
    left_sibling->n--;
}

void borrow_from_right_internal(Node *internal, Node *right_sibling, Node *parent, int parent_idx) {
    internal->keys[internal->n] = parent->keys[parent_idx];
    internal->children[internal->n + 1] = right_sibling->children[0];
    internal->n++;
    parent->keys[parent_idx] = right_sibling->keys[0];

    for (int i = 0; i < right_sibling->n - 1; i++) {
        right_sibling->keys[i] = right_sibling->keys[i + 1];
        right_sibling->children[i] = right_sibling->children[i + 1];
    }
    right_sibling->children[right_sibling->n - 1] = right_sibling->children[right_sibling->n];
    right_sibling->n--;
}

void merge_internal_nodes(Node *left, Node *right, Node *parent, int parent_idx) {
    left->keys[left->n] = parent->keys[parent_idx];
    left->children[left->n + 1] = right->children[0];
    left->n++;

    for (int i = 0; i < right->n; i++) {
        left->keys[left->n + i] = right->keys[i];
        left->children[left->n + i + 1] = right->children[i + 1];
    }
    left->n += right->n;

    for (int i = parent_idx; i < parent->n - 1; i++) {
        parent->keys[i] = parent->keys[i + 1];
        parent->children[i + 1] = parent->children[i + 2];
    }
    parent->n--;
    free(right);
}

bool delete_recursive(Node *node, int key) {
    if (node->is_leaf) {
        if (find_key_index(node, key) == -1) return false;
        remove_from_leaf(node, key);
        if (node == root) return false;
        return (node->n < MIN_KEYS);
    }

    int i = 0;
    while (i < node->n && key > node->keys[i]) i++;
    Node *child = node->children[i];

    bool underflow = delete_recursive(child, key);
    if (!underflow) return false;

    Node *left_sibling = (i > 0) ? node->children[i - 1] : NULL;
    Node *right_sibling = (i < node->n) ? node->children[i + 1] : NULL;

    if (left_sibling && left_sibling->n > MIN_KEYS) {
        if (child->is_leaf)
            borrow_from_left_leaf(child, left_sibling, node, i - 1);
        else
            borrow_from_left_internal(child, left_sibling, node, i - 1);
        return false;
    }

    if (right_sibling && right_sibling->n > MIN_KEYS) {
        if (child->is_leaf)
            borrow_from_right_leaf(child, right_sibling, node, i);
        else
            borrow_from_right_internal(child, right_sibling, node, i);
        return false;
    }

    if (left_sibling) {
        if (child->is_leaf)
            merge_leaf_nodes(left_sibling, child, node, i - 1);
        else
            merge_internal_nodes(left_sibling, child, node, i - 1);
        if (node == root && node->n == 0) {
            Node *old_root = root;
            root = root->children[0];
            free(old_root);
            return false;
        }
        return (node != root && node->n < MIN_KEYS);
    } else if (right_sibling) {
        if (child->is_leaf)
            merge_leaf_nodes(child, right_sibling, node, i);
        else
            merge_internal_nodes(child, right_sibling, node, i);
        if (node == root && node->n == 0) {
            Node *old_root = root;
            root = root->children[0];
            free(old_root);
            return false;
        }
        return (node != root && node->n < MIN_KEYS);
    }
    return false;
}

void delete_key(int key) {
    if (!root) return;
    delete_recursive(root, key);
}

/* Display */

void print_tree(Node *node, int level) {
    if (!node) return;
    for (int i = 0; i < level; i++) printf("  ");
    printf("[");
    for (int i = 0; i < node->n; i++) {
        printf("%d", node->keys[i]);
        if (i < node->n - 1) printf(",");
    }
    printf("]");
    if (node->is_leaf) {
        printf(" (leaf)");
        if (node->next) printf(" -> next leaf");
    } else {
        printf(" (internal)\n");
        for (int i = 0; i <= node->n; i++)
            print_tree(node->children[i], level + 1);
    }
    printf("\n");
}

void display_tree() {
    if (!root) {
        printf("Tree is empty.\n");
        return;
    }
    printf("\n\nB+ Tree structure:\n\n");
    print_tree(root, 0);
    printf("\n");
}

/* Main Menu */

int main() {
    int choice, key, value;
    while (1) {
        printf("\nB+ Tree Menu \n\n");
        printf("1. Insert key (with value)\n");
        printf("2. Delete key\n");
        printf("3. Search key\n");
        printf("4. Display tree\n");
        printf("5. Exit\n");
        choice = safe_get_int("\n\nEnter your choice: ");

        switch (choice) {
            case 1:
                key = safe_get_int("Enter key to insert: ");
                value = safe_get_int("Enter value for this key: ");
                insert(key, value);
                printf("\n\nInserted (%d, %d).\n", key, value);
                printf("-------------------------------------\n");
                break;
            case 2:
                key = safe_get_int("Enter key to delete: ");
                delete_key(key);
                printf("\n\nDeleted key %d (if it existed).\n", key);
                printf("-------------------------------------\n");
                break;
            case 3:
                key = safe_get_int("Enter key to search: ");
                value = search(root, key);
                if (value != -1)
                    printf("\n\nKey %d found with value %d.\n -------------------------------------\n", key, value);
                else
                    printf("\n\nKey %d not found.\n", key);
                    printf("----------------\n");
                break;
            case 4:
                display_tree();
                printf("-------------------------------------\n");
                break;
            case 5:
                printf("Exiting...\n");
                printf("-------------------------------------\n");
                return 0;
            default:
                printf("Invalid choice. Please enter a number between 1 and 5.\n");
        }
    }
    return 0;
}