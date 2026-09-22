/*
 * bplus_tree.c — B+ tree (order 4): int key -> int value, in C.
 *
 *   max keys per node : MAX_KEYS (3)  ->  max 4 children
 *   min keys per node : MIN_KEYS (1)  (all nodes except the root)
 *
 * Internal keys act as routing bounds, not required copies of leaf keys:
 *   child j holds keys  keys[j-1] <= key < keys[j]
 * so descent uses `key >= keys[i] -> go right`.
 * Leaves are linked via ->next, so range scans are one linear walk.
 *
 * Build:  gcc -std=c11 -Wall -Wextra -O2 -o bplus_tree data_structures/bplus_tree.c
 * Test:   ./bplus_tree --test     (randomized stress + invariant checks)
 * Demo:   ./bplus_tree            (interactive menu)
 */

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_KEYS 3
#define MIN_KEYS (MAX_KEYS / 2) /* = 1 */

typedef struct Node {
    bool is_leaf;
    int n;                      /* number of keys */
    int keys[MAX_KEYS];
    int values[MAX_KEYS];       /* leaves only */
    struct Node *children[MAX_KEYS + 1]; /* internal only */
    struct Node *next;          /* leaves only: next leaf in key order */
} Node;

static Node *root = NULL;

static Node *new_node(bool leaf) {
    Node *nd = (Node *)calloc(1, sizeof *nd);
    if (!nd) {
        fprintf(stderr, "bplus_tree: out of memory\n");
        exit(1);
    }
    nd->is_leaf = leaf;
    return nd;
}

static const Node *leftmost_leaf(const Node *nd) {
    while (nd && !nd->is_leaf)
        nd = nd->children[0];
    return nd;
}

/* ------------------------------------------------------------------ */
/* Search                                                              */
/* ------------------------------------------------------------------ */

/* Child branch for key: first i with key < keys[i]. */
static int child_index(const Node *nd, int key) {
    int i = 0;
    while (i < nd->n && key >= nd->keys[i])
        i++;
    return i;
}

bool bpt_search(int key, int *out_value) {
    if (!root)
        return false;
    const Node *nd = root;
    while (!nd->is_leaf)
        nd = nd->children[child_index(nd, key)];
    for (int i = 0; i < nd->n; i++) {
        if (nd->keys[i] == key) {
            if (out_value)
                *out_value = nd->values[i];
            return true;
        }
        if (nd->keys[i] > key)
            break;
    }
    return false;
}

/* ------------------------------------------------------------------ */
/* Insert                                                              */
/* ------------------------------------------------------------------ */

/* Insert into a leaf. Splits if full. Returns true and sets *promo / *right
 * when the leaf split. Duplicate keys update the value in place. */
static bool leaf_insert(Node *leaf, int key, int value, int *promo, Node **right) {
    int i = 0;
    while (i < leaf->n && leaf->keys[i] < key)
        i++;

    if (i < leaf->n && leaf->keys[i] == key) { /* duplicate -> update */
        leaf->values[i] = value;
        return false;
    }
    if (leaf->n < MAX_KEYS) {
        for (int j = leaf->n; j > i; j--) {
            leaf->keys[j] = leaf->keys[j - 1];
            leaf->values[j] = leaf->values[j - 1];
        }
        leaf->keys[i] = key;
        leaf->values[i] = value;
        leaf->n++;
        return false;
    }

    /* Full: sort into a temporary of MAX_KEYS+1 entries, split 2 | 2. */
    int tk[MAX_KEYS + 1], tv[MAX_KEYS + 1], cnt = 0;
    for (int j = 0; j < i; j++) { tk[cnt] = leaf->keys[j]; tv[cnt] = leaf->values[j]; cnt++; }
    tk[cnt] = key; tv[cnt] = value; cnt++;
    for (int j = i; j < leaf->n; j++) { tk[cnt] = leaf->keys[j]; tv[cnt] = leaf->values[j]; cnt++; }

    int mid = cnt / 2;
    leaf->n = mid;
    for (int j = 0; j < mid; j++) {
        leaf->keys[j] = tk[j];
        leaf->values[j] = tv[j];
    }

    Node *r = new_node(true);
    r->n = cnt - mid;
    for (int j = 0; j < r->n; j++) {
        r->keys[j] = tk[mid + j];
        r->values[j] = tv[mid + j];
    }
    r->next = leaf->next;
    leaf->next = r;

    *promo = r->keys[0];
    *right = r;
    return true;
}

/* Insert (promo, right) after child at index pos. Splits if full. */
static bool internal_insert(Node *nd, int pos, int promo, Node *right,
                            int *up_promo, Node **up_right) {
    if (nd->n < MAX_KEYS) {
        for (int j = nd->n; j > pos; j--)
            nd->keys[j] = nd->keys[j - 1];
        for (int j = nd->n + 1; j > pos; j--)
            nd->children[j] = nd->children[j - 1];
        nd->keys[pos] = promo;
        nd->children[pos + 1] = right;
        nd->n++;
        return false;
    }

    /* Full: rebuild from a temporary, then split. */
    int tk[MAX_KEYS + 1];
    Node *tc[MAX_KEYS + 2];
    int nk = 0, nc = 0;
    tc[nc++] = nd->children[0];
    for (int j = 0; j < nd->n; j++) {
        if (j == pos) {
            tk[nk++] = promo;
            tc[nc++] = right;
        }
        tk[nk++] = nd->keys[j];
        tc[nc++] = nd->children[j + 1];
    }
    if (pos == nd->n) {
        tk[nk++] = promo;
        tc[nc++] = right;
    }

    int mid = nk / 2; /* promote tk[mid] */
    nd->n = mid;
    for (int j = 0; j < mid; j++)
        nd->keys[j] = tk[j];
    for (int j = 0; j <= mid; j++)
        nd->children[j] = tc[j];

    Node *r = new_node(false);
    r->n = nk - mid - 1;
    for (int j = 0; j < r->n; j++)
        r->keys[j] = tk[mid + 1 + j];
    for (int j = 0; j <= r->n; j++)
        r->children[j] = tc[mid + 1 + j];

    *up_promo = tk[mid];
    *up_right = r;
    return true;
}

static bool insert_rec(Node *nd, int key, int value, int *promo, Node **right) {
    if (nd->is_leaf)
        return leaf_insert(nd, key, value, promo, right);
    int i = child_index(nd, key);
    if (!insert_rec(nd->children[i], key, value, promo, right))
        return false;
    return internal_insert(nd, i, *promo, *right, promo, right);
}

void bpt_insert(int key, int value) {
    if (!root) {
        root = new_node(true);
        root->keys[0] = key;
        root->values[0] = value;
        root->n = 1;
        return;
    }
    int promo;
    Node *right;
    if (insert_rec(root, key, value, &promo, &right)) {
        Node *nr = new_node(false);
        nr->n = 1;
        nr->keys[0] = promo;
        nr->children[0] = root;
        nr->children[1] = right;
        root = nr;
    }
}

/* ------------------------------------------------------------------ */
/* Delete                                                              */
/* ------------------------------------------------------------------ */

/* Remove keys[idx] and children[idx+1] from an internal node. */
static void remove_separator(Node *p, int idx) {
    for (int j = idx; j < p->n - 1; j++)
        p->keys[j] = p->keys[j + 1];
    for (int j = idx + 1; j < p->n; j++)
        p->children[j] = p->children[j + 1];
    p->n--;
}

/* Move one entry from the left sibling into children[i]. */
static void borrow_left(Node *p, int i) {
    Node *c = p->children[i], *s = p->children[i - 1];
    if (c->is_leaf) {
        for (int j = c->n; j > 0; j--) {
            c->keys[j] = c->keys[j - 1];
            c->values[j] = c->values[j - 1];
        }
        c->keys[0] = s->keys[s->n - 1];
        c->values[0] = s->values[s->n - 1];
        c->n++;
        s->n--;
        p->keys[i - 1] = c->keys[0];
    } else {
        for (int j = c->n; j > 0; j--)
            c->keys[j] = c->keys[j - 1];
        for (int j = c->n + 1; j > 0; j--)
            c->children[j] = c->children[j - 1];
        c->keys[0] = p->keys[i - 1];
        c->children[0] = s->children[s->n];
        c->n++;
        p->keys[i - 1] = s->keys[s->n - 1];
        s->n--;
    }
}

/* Move one entry from the right sibling (children[i+1]) into children[i]. */
static void borrow_right(Node *p, int i) {
    Node *c = p->children[i], *s = p->children[i + 1];
    if (c->is_leaf) {
        c->keys[c->n] = s->keys[0];
        c->values[c->n] = s->values[0];
        c->n++;
        for (int j = 0; j < s->n - 1; j++) {
            s->keys[j] = s->keys[j + 1];
            s->values[j] = s->values[j + 1];
        }
        s->n--;
        p->keys[i] = s->keys[0];
    } else {
        c->keys[c->n] = p->keys[i];
        c->children[c->n + 1] = s->children[0];
        c->n++;
        p->keys[i] = s->keys[0];
        for (int j = 0; j < s->n - 1; j++)
            s->keys[j] = s->keys[j + 1];
        for (int j = 0; j < s->n; j++)
            s->children[j] = s->children[j + 1];
        s->n--;
    }
}

/* children[i-1] absorbs children[i]; separator keys[i-1] is dropped. */
static void merge_with_left(Node *p, int i) {
    Node *l = p->children[i - 1], *c = p->children[i];
    if (c->is_leaf) {
        for (int j = 0; j < c->n; j++) {
            l->keys[l->n + j] = c->keys[j];
            l->values[l->n + j] = c->values[j];
        }
        l->n += c->n;
        l->next = c->next;
    } else {
        int base = l->n;
        l->keys[base] = p->keys[i - 1];
        for (int j = 0; j < c->n; j++)
            l->keys[base + 1 + j] = c->keys[j];
        for (int j = 0; j <= c->n; j++)
            l->children[base + 1 + j] = c->children[j];
        l->n = base + 1 + c->n;
    }
    free(c);
    remove_separator(p, i - 1);
}

/* children[i] absorbs children[i+1]; separator keys[i] is dropped. */
static void merge_with_right(Node *p, int i) {
    Node *c = p->children[i], *r = p->children[i + 1];
    if (c->is_leaf) {
        for (int j = 0; j < r->n; j++) {
            c->keys[c->n + j] = r->keys[j];
            c->values[c->n + j] = r->values[j];
        }
        c->n += r->n;
        c->next = r->next;
    } else {
        int base = c->n;
        c->keys[base] = p->keys[i];
        for (int j = 0; j < r->n; j++)
            c->keys[base + 1 + j] = r->keys[j];
        for (int j = 0; j <= r->n; j++)
            c->children[base + 1 + j] = r->children[j];
        c->n = base + 1 + r->n;
    }
    free(r);
    remove_separator(p, i);
}

/* Restore the occupancy invariant of children[i] after it shrank. */
static void fix_underflow(Node *p, int i) {
    Node *l = (i > 0) ? p->children[i - 1] : NULL;
    Node *r = (i < p->n) ? p->children[i + 1] : NULL;
    if (l && l->n > MIN_KEYS) {
        borrow_left(p, i);
        return;
    }
    if (r && r->n > MIN_KEYS) {
        borrow_right(p, i);
        return;
    }
    if (l)
        merge_with_left(p, i);
    else if (r)
        merge_with_right(p, i);
    /* No sibling: only possible at the root, collapsed by bpt_delete(). */
}

/* Returns true if nd underflowed (caller then fixes it). */
static bool delete_rec(Node *nd, int key) {
    if (nd->is_leaf) {
        int i = 0;
        while (i < nd->n && nd->keys[i] < key)
            i++;
        if (i >= nd->n || nd->keys[i] != key)
            return false; /* not present: nothing changed */
        for (int j = i; j < nd->n - 1; j++) {
            nd->keys[j] = nd->keys[j + 1];
            nd->values[j] = nd->values[j + 1];
        }
        nd->n--;
        return nd != root && nd->n < MIN_KEYS;
    }
    int i = child_index(nd, key);
    if (!delete_rec(nd->children[i], key))
        return false;
    fix_underflow(nd, i);
    return nd != root && nd->n < MIN_KEYS;
}

void bpt_delete(int key) {
    if (!root)
        return;
    delete_rec(root, key);
    while (root && !root->is_leaf && root->n == 0) {
        Node *old = root;
        root = root->children[0];
        free(old);
    }
    if (root && root->is_leaf && root->n == 0) {
        free(root);
        root = NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Range scan / statistics                                             */
/* ------------------------------------------------------------------ */

void bpt_foreach_range(int lo, int hi, void (*cb)(int key, int value, void *ud),
                       void *ud) {
    if (!root)
        return;
    const Node *nd = root;
    while (!nd->is_leaf)
        nd = nd->children[child_index(nd, lo)];
    while (nd) {
        for (int i = 0; i < nd->n; i++) {
            if (nd->keys[i] > hi)
                return;
            if (nd->keys[i] >= lo)
                cb(nd->keys[i], nd->values[i], ud);
        }
        nd = nd->next;
    }
}

long bpt_count(void) {
    long total = 0;
    for (const Node *nd = leftmost_leaf(root); nd; nd = nd->next)
        total += nd->n;
    return total;
}

/* ------------------------------------------------------------------ */
/* Invariant validation                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *msg; /* first failure description, NULL while OK */
    int line;        /* source line of the first failure */
    int leaf_depth;  /* -1 until the first leaf is seen */
} VCtx;

static void verr(VCtx *v, const char *msg, int line) {
    if (!v->msg) {
        v->msg = msg;
        v->line = line;
    }
}

/* Verify the subtree rooted at nd: key counts, sorted keys, key ranges
 * (lo <= key < hi), uniform leaf depth. Returns true if it holds >= 1 key. */
static bool vnode(VCtx *v, const Node *nd, long long lo, long long hi,
                  int depth) {
    if (!nd) {
        verr(v, "missing child pointer", __LINE__);
        return false;
    }
    if (nd->n > MAX_KEYS)
        verr(v, "node holds more than MAX_KEYS keys", __LINE__);
    if (nd != root && nd->n < MIN_KEYS)
        verr(v, "node holds fewer than MIN_KEYS keys", __LINE__);
    if (nd == root && nd->is_leaf && nd->n == 0)
        verr(v, "empty root leaf (should have been freed)", __LINE__);
    if (nd == root && !nd->is_leaf && nd->n == 0)
        verr(v, "internal root without keys (should have collapsed)", __LINE__);

    for (int j = 1; j < nd->n; j++)
        if (nd->keys[j] <= nd->keys[j - 1]) {
            verr(v, "keys not strictly increasing", __LINE__);
            break;
        }
    if (nd->n > 0) {
        if ((long long)nd->keys[0] < lo)
            verr(v, "key below subtree range", __LINE__);
        if ((long long)nd->keys[nd->n - 1] >= hi)
            verr(v, "key above subtree range", __LINE__);
    }

    if (nd->is_leaf) {
        if (v->leaf_depth < 0)
            v->leaf_depth = depth;
        else if (v->leaf_depth != depth)
            verr(v, "leaves at different depths", __LINE__);
        return nd->n > 0;
    }

    bool nonempty = false;
    for (int j = 0; j <= nd->n; j++) {
        long long clo = (j > 0) ? nd->keys[j - 1] : lo;
        long long chi = (j < nd->n) ? nd->keys[j] : hi;
        bool e = vnode(v, nd->children[j], clo, chi, depth + 1);
        if (j == 0)
            nonempty = e;
        else if (!e)
            verr(v, "empty child subtree", __LINE__);
        /* Separators are routing bounds: child j >= keys[j-1] is already
         * enforced by that child's own range check; equality with the
         * child's current minimum is NOT required (deleting a separator
         * key from the leaves legitimately leaves it behind). */
    }
    return nonempty;
}

/* In-order leaf walk: every leaf must be linked exactly once, in order. */
static void vchain(VCtx *v, const Node *nd, const Node **prev) {
    if (!nd)
        return;
    if (nd->is_leaf) {
        if (*prev) {
            if ((*prev)->next != nd)
                verr(v, "leaf chain broken", __LINE__);
            if ((*prev)->n > 0 && nd->n > 0 &&
                nd->keys[0] <= (*prev)->keys[(*prev)->n - 1])
                verr(v, "leaf chain not sorted", __LINE__);
        }
        *prev = nd;
        return;
    }
    for (int j = 0; j <= nd->n; j++)
        vchain(v, nd->children[j], prev);
}

/* Returns NULL when the tree is valid, otherwise a description of the
 * first problem found. */
const char *bpt_validate(void) {
    if (!root)
        return NULL;
    VCtx v = {NULL, 0, -1};
    vnode(&v, root, LLONG_MIN, LLONG_MAX, 0);
    const Node *prev = NULL;
    vchain(&v, root, &prev);
    if (prev && prev->next)
        verr(&v, "leaf chain has an extra leaf", __LINE__);
    return v.msg;
}

/* ------------------------------------------------------------------ */
/* Display                                                             */
/* ------------------------------------------------------------------ */

static void print_node(const Node *nd, int depth) {
    for (int i = 0; i < depth; i++)
        fputs("  ", stdout);
    putchar('[');
    for (int i = 0; i < nd->n; i++) {
        if (nd->is_leaf)
            printf("%d=%d", nd->keys[i], nd->values[i]);
        else
            printf("%d", nd->keys[i]);
        if (i + 1 < nd->n)
            fputs(", ", stdout);
    }
    printf("] %s", nd->is_leaf ? "leaf" : "internal");
    if (nd->is_leaf && nd->next)
        fputs(" ->", stdout);
    putchar('\n');
    if (!nd->is_leaf)
        for (int i = 0; i <= nd->n; i++)
            print_node(nd->children[i], depth + 1);
}

void bpt_print(void) {
    if (!root) {
        puts("(empty tree)");
        return;
    }
    print_node(root, 0);
}

/* ------------------------------------------------------------------ */
/* Test suite (--test)                                                 */
/* ------------------------------------------------------------------ */

static int g_fail = 0;

static void check(bool cond, const char *what, int line) {
    if (!cond) {
        printf("FAIL (line %d): %s\n", line, what);
        g_fail++;
    }
}
#define CHECK(c) check((c), #c, __LINE__)

static bool vok(void) {
    const char *e = bpt_validate();
    if (e) {
        printf("FAIL validate: %s\n", e);
        g_fail++;
        return false;
    }
    return true;
}

typedef struct {
    long n;
} Counter;

static void cb_count(int k, int v, void *ud) {
    (void)k;
    (void)v;
    ((Counter *)ud)->n++;
}

typedef struct {
    const unsigned char *has;
    const int *val;
    int idx;  /* next model index expected from the scan */
    int off;  /* key = index - off */
    long space;
    bool bad;
} SweepCtx;

static void cb_sweep(int k, int v, void *ud) {
    SweepCtx *s = (SweepCtx *)ud;
    while (s->idx < s->space && !s->has[s->idx])
        s->idx++;
    int r = k + s->off;
    if (s->idx >= s->space || r != s->idx || !s->has[r] || s->val[r] != v)
        s->bad = true;
    else
        s->idx++;
}

static void test_sequential(void) {
    enum { N = 80 };
    for (int i = 1; i <= N; i++)
        bpt_insert(i, i * 10);
    CHECK(vok());
    CHECK(bpt_count() == N);

    int out = 0;
    CHECK(bpt_search(37, &out) && out == 370);
    CHECK(!bpt_search(0, &out));
    CHECK(!bpt_search(N + 1, &out));

    bpt_insert(40, 999); /* duplicate key updates the value */
    CHECK(bpt_search(40, &out) && out == 999);
    CHECK(bpt_count() == N);

    Counter c = {0};
    bpt_foreach_range(10, 20, cb_count, &c);
    CHECK(c.n == 11);

    for (int i = 1; i <= N; i++) {
        bpt_delete(i);
        if (!vok())
            break;
    }
    CHECK(root == NULL);
    CHECK(bpt_count() == 0);

    /* Same, inserting ascending and deleting descending. */
    for (int i = 1; i <= N; i++)
        bpt_insert(i, i);
    CHECK(vok());
    for (int i = N; i >= 1; i--) {
        bpt_delete(i);
        if (!vok())
            break;
    }
    CHECK(root == NULL);
}

static void test_random(void) {
    enum { SPACE = 400, OPS = 40000, OFF = SPACE / 2 };
    static unsigned char has[SPACE];
    static int val[SPACE];
    memset(has, 0, sizeof has);
    srand(20260922);

    for (int op = 1; op <= OPS; op++) {
        int r = rand() % SPACE, key = r - OFF;
        if (rand() % 100 < 62) {
            int v = rand() % 1000000;
            bpt_insert(key, v);
            has[r] = 1;
            val[r] = v;
        } else {
            bpt_delete(key);
            has[r] = 0;
        }

        int out = 0;
        bool f = bpt_search(key, &out);
        if (f != (has[r] != 0) || (f && out != val[r])) {
            printf("FAIL: search mismatch at op %d (key %d)\n", op, key);
            g_fail++;
            return;
        }

        if (op % 500 == 0) {
            const char *e = bpt_validate();
            if (e) {
                printf("FAIL: validate at op %d: %s\n", op, e);
                g_fail++;
                return;
            }
            for (int j = 0; j < SPACE; j++) {
                int k2 = j - OFF, o2 = 0;
                bool f2 = bpt_search(k2, &o2);
                if (f2 != (has[j] != 0) || (f2 && o2 != val[j])) {
                    printf("FAIL: full sweep at op %d (key %d)\n", op, k2);
                    g_fail++;
                    return;
                }
            }
            long pc = 0;
            for (int j = 0; j < SPACE; j++)
                pc += has[j];
            if (bpt_count() != pc) {
                printf("FAIL: chain count %ld != model %ld at op %d\n",
                       bpt_count(), pc, op);
                g_fail++;
                return;
            }
            SweepCtx sc = {has, val, 0, OFF, SPACE, false};
            bpt_foreach_range(-OFF, SPACE - 1 - OFF, cb_sweep, &sc);
            if (sc.bad) {
                printf("FAIL: range sweep at op %d\n", op);
                g_fail++;
                return;
            }
        }
    }
    CHECK(vok());
}

static void test_edge(void) {
    /* Start from a clean slate: the random test leaves keys behind.
     * Draining by repeatedly deleting the minimum exercises merges too. */
    while (root) {
        const Node *lf = leftmost_leaf(root);
        if (!lf || lf->n == 0)
            break; /* defensive: would otherwise loop forever */
        bpt_delete(lf->keys[0]);
    }
    CHECK(root == NULL);

    int out = 0;
    CHECK(!bpt_search(1, &out)); /* search in empty tree */
    bpt_delete(1);               /* delete in empty tree: no crash */
    CHECK(root == NULL);

    bpt_insert(-7, 70);
    bpt_insert(-7, 71); /* duplicate on a fresh root leaf */
    CHECK(bpt_search(-7, &out) && out == 71);
    CHECK(bpt_count() == 1);
    bpt_delete(-7);
    CHECK(root == NULL);

    for (int i = 0; i < 30; i++)
        bpt_insert(i * 3, i);
    CHECK(vok());
    for (int i = 0; i < 30; i++)
        bpt_delete(i * 3 + 1); /* absent keys: no structural change */
    CHECK(vok());
    CHECK(bpt_count() == 30);
    for (int i = 0; i < 30; i++)
        bpt_delete(i * 3);
    CHECK(root == NULL);
}

static int run_tests(void) {
    test_sequential();
    test_random();
    test_edge();
    if (g_fail) {
        printf("\n%d FAILURE(S)\n", g_fail);
        return 1;
    }
    puts("All B+ tree tests passed "
         "(sequential, 40k randomized ops with invariant checks, edges).");
    return 0;
}

/* ------------------------------------------------------------------ */
/* Interactive menu                                                    */
/* ------------------------------------------------------------------ */

static int read_int(const char *prompt) {
    int val;
    char buf[64];
    for (;;) {
        printf("%s", prompt);
        fflush(stdout);
        if (!fgets(buf, sizeof buf, stdin)) {
            puts("\nEOF — exiting.");
            exit(0);
        }
        if (sscanf(buf, "%d", &val) == 1)
            return val;
        puts("Invalid input, please enter an integer.");
    }
}

static void range_print_cb(int k, int v, void *ud) {
    (void)ud;
    printf("  %d -> %d\n", k, v);
}

static int menu(void) {
    for (;;) {
        puts("\n--- B+ tree (order 4) ---");
        puts("1) insert   2) delete   3) search   4) print");
        puts("5) range    6) validate  7) exit");
        switch (read_int("> ")) {
        case 1: {
            int k = read_int("key: ");
            int v = read_int("value: ");
            bpt_insert(k, v);
            printf("inserted %d -> %d (count=%ld)\n", k, v, bpt_count());
            break;
        }
        case 2: {
            int k = read_int("key: ");
            int out = 0;
            bool had = bpt_search(k, &out);
            bpt_delete(k);
            if (had)
                printf("deleted %d\n", k);
            else
                printf("%d was not present\n", k);
            break;
        }
        case 3: {
            int k = read_int("key: ");
            int out = 0;
            if (bpt_search(k, &out))
                printf("found: %d -> %d\n", k, out);
            else
                printf("%d not found\n", k);
            break;
        }
        case 4:
            bpt_print();
            printf("count=%ld\n", bpt_count());
            break;
        case 5: {
            int lo = read_int("lo: ");
            int hi = read_int("hi: ");
            Counter c = {0};
            printf("keys in [%d, %d]:\n", lo, hi);
            bpt_foreach_range(lo, hi, range_print_cb, NULL);
            bpt_foreach_range(lo, hi, cb_count, &c);
            printf("(%ld keys)\n", c.n);
            break;
        }
        case 6: {
            const char *e = bpt_validate();
            puts(e ? e : "invariants OK");
            break;
        }
        case 7:
            return 0;
        default:
            puts("invalid choice");
        }
    }
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--test") == 0)
        return run_tests();
    puts("B+ tree order 4  (run with --test for the self-check suite)");
    return menu();
}
