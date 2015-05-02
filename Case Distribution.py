# coding=utf-8
"""This program seeks to evaluate a structure-based case realization algorithm
by comparing it to other theories (and non-theories) of case distribution by
testing them on the Icelandic Parsed Historical Corpus
(https://linguist.is/icelandic_treebank/).
Author: Miles Calabresi
Project: Yale University Senior Thesis in Linguistics
Spring 2015"""

__author__ = 'Miles Calabresi'

import sys
import re
import random
# import antigravity
from nltk.tree import ParentedTree


##############################################################################
# My functions
# NOTE: I use 'st' here to refer to all subtree variables;
# in the main body, I will instead use 'node'
# That is, do not use variable 'node' here -- it is protected; similarly,
# avoid use of 'st' in the program body.
##############################################################################


def ns_dominates(a, b):
    """Simple predicate to tell if a non-strictly dominates b
    :param a: any node in tree
    :param b: any node in the same tree
    """
    # Check that a and b lie in the same tree
    assert a.root() == b.root()
    if a == b:
        return True
    elif b == b.root():
        return False
    else:
        return ns_dominates(a, b.parent())


# Note: this is strict domination
def dominates(a, b):
    """Predicate to tell if node a strictly dominates node b
    :param a: any node in a tree
    :param b: any node in the same tree
    """
    assert a.root() == b.root()
    if a == b:
        return False
    else:
        return ns_dominates(a, b)


def precedes(a, b):
    """
    Tells whether node a precedes node b. Returns false if they the same node.
    :param a: any node in a tree
    :param b: any node in the same tree
    :return: Boolean corresponding to whether a comes before b in a left-to-
    right (depth-first) traversal of the tree
    """
    assert a.root() == b.root()
    lca = find_least_common_ancestor(a, b)
    if a == b:
        return False
    for n in lca.subtrees():
        if n == a:
            return True
        elif n == b:
            return False


def c_commands(a, b):
    """
    :param a: any node in a tree
    :param b: any node in the same tree
    :return: Boolean value telling whether a c-commands b
    """
    assert a.root() == b.root()  # Make sure they're in the same tree.
    assert ns_dominates(a.root(), b)
    if ns_dominates(a, b) or ns_dominates(b, a):
        return False
    lca = find_least_common_ancestor(a, b)
    while a != lca:
        a = a.parent()
        if len(a) > 1:  # that is, if a has more than one child
            assert dominates(a, b) == ns_dominates(a, b)
            return dominates(a, b)
    # if we are here, then something has gone wrong: the LCA of a and b
    # has no branching children between a and it, but neither a nor b
    # dominates one another
    print('Error: bad least common ancestor of', a, 'and', b)
    sys.exit(1)


def spec_c_commands(a, b, sym):
    """
    Modified predicate to determine whether a node (a)symmetrically c-commands
    another node
    :param a: any node in a tree
    :param b: any node in the same tree
    :param sym: a boolean to determine asymmetric or symmetric c-commanding
    """
    if sym:
        return c_commands(a, b) and c_commands(b, a)
    else:
        return c_commands(a, b) and not c_commands(b, a)


def is_noun(st, ignore_dubs=True, include_quants=True):
    """
    A Predicate to tell whether a given node in the tree is a noun
    and thus able to be marked with case, possibly ignoring certain kinds of
    nouns.
    :param st: a node in a tree
    :param ignore_dubs: a flag to tell whether or not to return true for
    three kinds of nouns that I consider duplicates: appositives, low
    coordinated/conjoined nouns, and right proper nouns. All of these
    categories are assumed to receive case in the same way as their partners
    (the noun modified by the appositive, the other conjunct(s) and the first/
    leftmost proper noun in the sequence). Therefore, I choose to ignore them
    so they aren't double-counted.
    :param include_quants: a flag to tell whether to return true for
    quantifiers that act as pronouns (not adjectives)
    :return: Boolean
    """
    if isinstance(st, str):
        # Don't count Null nodes, which show up as strings not trees.
        return False
    # the @ symbol is my placeholder character for "unmarked"
    if re.match('(N(PR)?S?|W?PRO)-[NADG@]', st.label()):
        if ignore_dubs:
            p = st.parent().label()
            ls = ''
            if st.left_sibling() is not None:
                ls = st.left_sibling().label()
            if p[:5] == 'CONJP' or p[:6] == 'NP-PRN' or ls[:3] == 'NPR' or \
                    ('CONJ' in [child.label() for child in st.parent()] and
                     (st.right_sibling() is None or
                      st.right_sibling().label() != 'CONJ')):
                return False
        return True
    elif include_quants:
        if (re.match('W?QR?S?(\+NUM)?-[NADG@]', st.label()[:2]) or
           st.label()[:3] == 'ONE'):
            for child in st.parent():
                if is_noun(child, ignore_dubs, include_quants=False) or \
                   child.label()[:2] == 'NP':
                    return False
            return True
    return False


def is_verb(st):
    """
    A Predicate to tell whether a given node in a tree is a verb.
    :param st: a node in some tree
    :return: Boolean corresponding to whether the node is a verb
    """
    assert type(st) != str
    return type(st[0]) == str and (st.label()[:2] == 'VB' or
                                   st.label()[:2] == 'VA' or
                                   st.label()[:2] == 'RD' or
                                   # include preps for quirky purposes
                                   st.label() == 'P')


def is_unmarked(word):
    """
    Predicate to determine whether a given noun (or pronoun) is currently
    unmarked for case
    :rtype : Boolean
    :param word: any noun head node in a tree
    :return: Boolean value corresponding to if word is not case-marked
    """
    if is_noun(word):
        return word.label()[-1] == '@'
    else:
        sys.exit('Error: The node ' + word +
                 ' inputted is not a noun; cannot check case marking.')


def find_max_proj(n_head):
    """
    Find the maximal projection of a given noun head, ignoring appropriate
    intermediate layers, and interpreting any intermediate projection in its
    base position as necessary.
    :param n_head: A noun head in a tree
    :return:
    """
    assert is_noun(n_head)
    # These labels are empirically the ones that intervene between N heads
    # and their maximum projections. NP-internal possessors are a known
    # pitfall, so we stop specifically at them.
    prev = find_base_pos(n_head)
    n_head = find_base_pos(n_head)
    while n_head.parent() is not None and n_head.label()[:6] != 'NP-POS' and\
            ((n_head.parent().label()[:2] == 'NP') or
             (n_head.parent().label()[:2] == 'NX') or
             (n_head.parent().label()[:2] == 'QP') or
             (n_head.parent().label()[:3] == 'WNP') or
             (n_head.parent().label()[:5] == 'CONJP') or
             (n_head.parent().label()[:4] == 'CODE')):
        prev = find_base_pos(n_head)
        n_head = find_base_pos(n_head.parent())
    if n_head.label()[:5] == 'CONJP' or n_head.label()[:4] == 'CODE':
        n_head = prev
    return n_head


def find_base_pos(word, a_mvmt=False):
    """
    Find and return the base position of a constituent (defined as its current
    position if it hasn't been moved)
    :param word: a node in the tree
    :param a_mvmt: flag to toggle whether to move down to base position of
    A-movement and A'-movement or just A'-movement (default)
    :return: a different node corresponding to the base position of word
    """
    if len(word.label()) < 2:
        return word
    num = word.label()[-2:]
    if not re.match('-\d', num):
        return word
    found = []
    traces = ['*ICH*']
    if a_mvmt:
        traces.extend(['*T*', '*'])
    traces = [s + num for s in traces]
    for st in word.root().subtrees():
        if st[0] in traces:
            found.append(st)
    if len(found) == 1:
        return found[0]
    else:
        if verify('Error finding unique base position for' + word.label() +
                  'in tree\n' + str(word.root()[1]) + '\nFound ' + str(found)):
            return word


def find_surf_pos(word):
    """
    Find the surface position of a moved constituent, or its current position
    if the constituent is not moved
    :param word: a node in the tree
    :return: a node corresponding to the surface position of word
    """
    num = word[0][-1]
    if re.match('\d', num) is None:
        return word
    found = []
    for st in word.root().subtrees():
        if st.label()[-2:] == '-' + num and re.match('W?NP', st.label()[:3]):
            found.append(st)
    if len(found) == 1:
        return found[0]
    else:
        if verify('Error finding unique surface position for ' + str(word) +
                  ' in tree\n' + str(word.root()) + '\nFound ' + str(found)):
            return word
        else:
            sys.exit(1)


def mark(word, case):
    """
    Given a node in a tree and a case as a string, mark that node (which should
    be a noun) as the given case, and return the *root* of a new tree with that
    change.
    :param word: a node in the tree
    :param case: a string describing a given case
    :return:
    """
    if not is_noun(word):
        sys.exit('Trying to mark non-noun.')
    if case != '@' and not is_unmarked(word):
        sys.exit('Attempted overwrite of case.')
    nt = ParentedTree.fromstring(str(word.root()))
    nt[word.treeposition()].set_label(word.label()[:-1] + case)
    return nt


def mark_all(tree, case):
    """
    a function to mark all noun heads in a given tree the same case
    :param tree: any tree
    :param case: any case marking (as a string)
    :return: the root of the tree, but with all unmarked nouns marked in the
    given case, unless the given case is @, then all nouns, marked or not,
    are assigned @
    """
    rt = tree.root()
    for st in tree.root().subtrees():
        if is_noun(st):
            rt = mark(rt[st.treeposition()], case)
    return rt


def strip_case(tree):
    """
    A function to strip a given tree of case and return the unmarked tree.
    :param tree: any tree
    :return: the same tree but with case removed from the noun heads
    """
    return mark_all(tree, '@')


def count_case_freq(tree, counter):
    """
    A function to count the frequencies in a given tree of each case in a
    given counter.
    :param tree: an NLTK syntax tree
    :param counter: any existing counts to be added to
    :return:
    """
    for word in tree.subtrees():
        if is_noun(word):
            try:
                counter[word.label()[-1]] += 1
            except KeyError:
                print('Error: bad case ' + word.label()[-1] + ' on ' +
                      str(word))
    return counter


def score_tree(corpus, test, existing=None):
    """
    :param corpus: a tree directly from the corpus; the "correctly" marked one
    :param test: the tree to be tested against corpus for correct case
    :param existing: the existing scorecard (if any) for (incorrect) case
    :return:
    """
    # My convention is that the first letter is the corpus case; the second is
    # the case marked by the algorithm.
    if existing is None:
        existing = {'NN': 0, 'NA': 0, 'ND': 0, 'NG': 0,
                    'AN': 0, 'AA': 0, 'AD': 0, 'AG': 0,
                    'DN': 0, 'DA': 0, 'DD': 0, 'DG': 0,
                    'GN': 0, 'GA': 0, 'GD': 0, 'GG': 0,
                    'N@': 0, 'A@': 0, 'D@': 0, 'G@': 0}
    corp_nodes = [st for st in corpus.subtrees()]
    test_nodes = [st for st in test.subtrees()]
    assert len(corp_nodes) == len(test_nodes)

    pairs = [(corp_nodes[i].label(), test_nodes[i].label())
             for i in range(len(test_nodes)) if is_noun(test_nodes[i])]

    for pair in pairs:
        existing[pair[0][-1] + pair[1][-1]] += 1
    return existing


def print_counts(counts, source):
    """
    print the confusion matrix for counts of case marked correctly and
    incorrectly from a counter/scorecard
    :param counts: an iterable containing all of the (in)correct case marks
    :param source: a string describing the source/document where the marking
    took place and where the counts were drawn from
    :return: none; prints out counts
    """
    print('Total counts for each case in', source + ':')
    print('\t', counts)
    print('Frequencies:')
    if sum(counts.values()) > 0:
        for key in sorted(counts.keys(), key=lambda x: 'NADG@'.index(x)):
            print('\t', (key, '{:.3%}'.format(counts[key] * 1. /
                                              sum(counts.values()))))
    else:
        print('No counts in', counts, 'from', source)


def pp_score(card, mat=True, incl_unmarked=True):
    """
    prints the scorecard's results and percentages legibly
    :param card: a card consisting of two-letter counts corresponding to
    instances of (in)correct case marking
    :param mat: toggle whether to print the card in matrix (True) or list
    format
    :param incl_unmarked: toggle whether to count unmarked nouns in the
    calculation of the recall score, or just ones marked incorrectly
    :return: non; prints out results
    """
    right = 0
    unmarked = 0
    marked_wrong = 0

    # Confusion matrix version
    if mat:
        ln = 1 + 13  # length of all three next lines
        spc = ''
        for i in range(ln):
            spc += ' '
        print('\nCase assigned\nby algorithm:'
              '\t\tN\t\t\tA\t\t\tD\t\t\tG\t\t\t@')
        print('Correct case: ', end='')
        for case in 'NADG':
            if case != 'N':
                print(spc, end='')
            print(case + '\t\t', end='')
            for case2 in 'NADG@':
                num = str(card[case + case2])
                if len(num) < 3:
                    num += '  '
                print(num, '\t\t', end='')
            print()
    for item in sorted(iter(card)):
        if item[0] != item[1]:
            if not mat:
                # Print simple list, not full matrix
                print('Case ' + item[0] + ' mistaken for ' + item[1] +
                      ' by algorithm: ' + str(card[item]))
            if item[1] == '@':
                unmarked += card[item]
            else:
                marked_wrong += card[item]
        else:
            # Print simple list, not full matrix
            if not mat:
                print('Case ' + item[0] + ' marked correctly: '
                      + str(card[item]))
            right += card[item]
    if right == 0:  # for scenarios when wrong = 0
        # First excludes unmarked nouns, second includes them
        pct = [0.000, 0.000]
    else:
            pct = [right * 1. / (marked_wrong + right),
                   right * 1. / (unmarked + marked_wrong + right)]

    print('Total marked correctly:', str(right),
          '\t({:.3%}; {:.3%})'.format(pct[0], pct[1]))
    print('Total left unmarked:', unmarked)
    print('Total marked incorrectly:', marked_wrong)
    print('Total wrong:', unmarked + marked_wrong)

    print('\nSTATISTICS BY INDIVIDUAL CASE')
    cases = 'NADG'
    if incl_unmarked:
        cases += '@'
    for case in 'NADG':
        relevant = float(sum([card[case + case2] for case2 in cases]))
        selected = float(sum([card[case2 + case] for case2 in 'NADG']))
        correct = float(card[case + case])
        if selected > 0:
            precision = correct / selected
        else:
            precision = 0
        if relevant > 0:
            recall = correct / relevant
        else:
            recall = 0
        print('Case:', case)
        print('\tPrecision: {:.3%}'.format(precision))
        print('\t   Recall: {:.3%}'.format(recall))
        print('\t  F-score: {:.3%}'.format(f_score(precision, recall)))


def f_score(precision, recall, beta=1):
    """
    Calculate the f-score for a given beta (default =1) from the precision
    and recall values.
    :param precision: value of precision (true positives over actual correct
    answers)
    :param recall: value of recall (true positives over all selected answers)
    :param beta: weight of recall with respect to precision
    :return:
    """
    if precision + recall == 0:
        return 0
    else:
        return (1 + beta * beta) * float(precision * recall) / \
            float(beta * beta * precision + recall)


def verify(msg=''):
    """
    Verify that it's okay to continue or quit; used to quit if a possible
    error arises.
    :param msg: an optional text message to explain what needs to be verified
    :return: nothing -- either continue or quit
    """
    if not safe_mode:
        return True

    resp = input(msg + "\nContinue?\n").lower()
    while True:
        if resp == 'y' or resp == 'yes':
            return True
        elif resp == 'n' or resp == 'no':
            sys.exit(-1)
        else:
            resp = input("I don't understand. Should we continue?\n")


# ########## Algorithms ###########
# #################################
# Randomized algorithm

def choose_rand_weighted(freqs):
    """
    given a dictionary, choose a random key from it, weighted by the
    frequencies (values) of each key
    :param freqs: a dictionary whose values are the counts of the keys
    :return: a key from the dictionary
    """
    lst = []
    for item in freqs.keys():
        for i in range(freqs[item]):
            lst.append(item)
    return random.choice(lst)


def mark_random(tree, freqs=None):
    """
    function to mark all nouns with a random case according to a given
    frequency distribution or uniform distribution if none given
    :param tree: any tree
    :param freqs: a dictionary of cases and their relative frequencies
    :return: the tree with all previously unmarked nouns marked with a
    randomly chosen case
    """
    if freqs is None:
        freqs = {'N': 1, 'A': 1, 'D': 1, 'G': 1}
    rt = tree.root()
    for st in tree.root().subtrees():
        if is_noun(st) and is_unmarked(st):
            rt = mark(rt[st.treeposition()], choose_rand_weighted(freqs))
    return rt


def find_head(np):
    """
    Search descendants of a given noun phrase layer for the head of the NP.
    :param np: a NP projection in a tree
    :return: the N head of that NP (in that same tree)
    """
    if is_noun(np):
        return np
    if np.label()[:2] not in ['Q-', 'NP'] and \
       np.label()[:3] not in ['WNP', 'ONE']:
        print('Bad noun phrase', np.label(), 'in find_head function:', np,
              flush=True)
        assert np.label()[:2] == 'NP' or np.label()[:3] == 'WNP'
    for child in np:
        if is_noun(child):
            return child
    for child in np:
        if not isinstance(child, str) and re.match('W?NP', child.label()[:3]) \
                and not re.match('-POS', child.label()):
            return find_head(child)
    verify('No N head child of NP ' + str(np))
    return np


def find_func(n_head, func):
    """
    Search ancestors of a given noun head for a function-marked NP layer,
    stopping at anything but intermediate noun layers, conjunction phrases, or
    non-structural layers.
    :param n_head: a noun head node in any tree
    :param func: a string describing the grammatical function of the sought NP
    :return: a node in the same tree corresponding to NP with the given
    function that contains the given N head, or None if none found
    """
    # find_base_pos is benign if the NP/NX parent wasn't (A'-)moved
    np = find_base_pos(find_max_proj(n_head))
    if re.match('W?PP', np.parent().label()[:3]) and func == 'PPOBJ':
        return True
    elif np.label()[:3+len(func)] == 'NP-' + func or np.label()[:4+len(func)]\
            == 'WNP-' + func:
        return np
    else:
        verify('No function found in' + str(np) + '\n\n' + str(n_head))
        return None


def find_least_common_ancestor(a, b):
    """
    find the least common ancestor of nodes a and b in a tree
    :param a: any node in a tree
    :param b: any node in the same tree
    :return: a node in the tree that is the least common ancestor of a and b
    """
    assert a.root() == b.root()
    while not ns_dominates(b, a):
        b = b.parent()
    return b


def crosses(st, ancestor, layers):
    """
    determine whether one crosses a layer of type layer while going from
    the node st to the node ancestor (or vice versa)
    :param st: any node in a tree
    :param ancestor: an ancestor of st, including possibly st itself
    :param layers: the types of node to test for, as strings, e.g. CP and
    :return:
    """
    assert st.root() == ancestor.root()
    assert ns_dominates(ancestor, st)
    if st == ancestor:
        return False
    st = st.parent()
    while st != ancestor:
        for layer in layers:
            if st.label()[:len(layer)] == layer:
                return True
        st = st.parent()
    return False


def same_domain(a, b):
    """
    determine whether two nodes are in the same case marking domain by testing
    whether one crosses a CP/TP/IP (as dictated by your theory) getting from
    least common ancestor of A and B to either A or to B.
    :param a: any node in a tree
    :param b: any node in the same tree
    :return: a boolean telling whether nodes a and b are in the same case-
    marking domain
    """
    assert a.root() == b.root()
    if a == b:
        return True
    lca = find_least_common_ancestor(a, b)
    return not (crosses(a, lca, ['CP']) or
                crosses(b, lca, ['CP']))


def mark_args(verb, case_frame, correct_tree):
    """
    Take a lexical item (verb, or prepositions) and find the
    arguments (like subject, direct object complement) that it lexically
    governs. These will be marked for lexical case.
    Locate the various arguments of a verb
        o	Subject – search verb's ancestors' siblings (up to CP) for an
            NP-SBJ node
        o	Direct Object – search subtree headed by verb’s sister for NP-OB1
            (stopping at CP, NP, PP)
        o	Indirect Object – search verb's sister’s subtree for NP-OB2 or
            NP-OB3 (stop at same nodes as)
    :param verb: a verb node in a tree
    :param case_frame: a three-letter string describing which case to
            mark each of the three arguments of the verb
    :param correct_tree: the tree with the correct case markings, used
    to keep track of where this function fails and succeeds
    """
    if len(case_frame) != 3:
        print('Error with lexical case frame', '"' + case_frame + '" of',
              verb, 'in\n', verb.root())
        sys.exit(1)

    if not is_verb(verb):
        print('Error in mark_args function:', verb, 'is not a verb.')
        sys.exit(1)

    # For each argument (subject, direct object, or indirect object), if the
    # verb governs the argument's case, try to find the given argument.
    # To do this, we search the tree for NP-ARG nodes and find out which, if
    # any, c-command the verb. If there is a unique one, mark it.
    # For objects (both direct and indirect), we do the same, but instead we
    # verify that the verb c-commands the object, rather than the other way
    # around. In all cases, we check for moved constituents.
    # Repeat for each kind of argument.

    tree = verb.root()
    arg_types = [['subject', 'SBJ', 'SBJ'],
                 ['indirect object', 'OB2', 'OB3'],
                 ['direct object', 'OB1', 'OB1']]
    # For i = 1 (indirect objects), we need to check if it is OB2 or OB3
    # For i = 0 or 2, we just check the first condition again (hence the
    # redundant third item in those sub-lists.

    for i in range(len(case_frame)):
        if case_frame[i] != '-':
            if verb.root() != tree:
                verb = tree[verb.treeposition()]
            found = []
            if i == 0:  # looking for subjects
                cc_cond = lambda x, y: c_commands(x, y)
            else:  # looking for some kind of object
                cc_cond = lambda x, y: c_commands(y, x)

            for st in tree.subtrees():
                # Search for an NP that stands in the appropriate c-command
                # relation to the given verb/prep head within the same domain.
                # For verbs, we must find the right kind of argument like -SBJ.
                # For prepositions, it is sufficient to find just an NP.
                if (st.label()[:6] == 'NP-' + str(arg_types[i][1]) or
                    st.label()[:7] == 'WNP-' + str(arg_types[i][1]) or
                    st.label()[:6] == 'NP-' + str(arg_types[i][2]) or
                    st.label()[:7] == 'WNP-' + str(arg_types[i][2]) or
                    ((st.label()[:2] == 'NP' or st.label()[:3] == 'WNP')
                     and verb.label() == 'P')) and cc_cond(st, verb) and \
                        same_domain(st, verb):
                    if st[0][-7:-2] == '*ICH*' or st[0][-5:-2] == '*T*' or \
                            st[0][:2] == '*-':
                        st = find_surf_pos(st)
                    found.append(st)

            # Now keep statistics of whether each verb's arguments
            # were found and marked correctly or incorrectly.
            verb_lemma = verb[0][verb[0].index('-') + 1:] + ':' + str(i + 1) +\
                '-' + case_frame[i]
            if verb_lemma not in lex_verbs:
                # Format: [correct, {Counts for what it should have been},
                # [found none, found too many] ]
                lex_verbs[verb_lemma] = [0, {'N': 0, 'A': 0, 'D': 0, 'G': 0},
                                         [0, 0]]

            if len(found) == 0:
                counts_by_function[i][2] += 1
                lex_verbs[verb_lemma][2][0] += 1
                verify('No ' + arg_types[i][0] + ' of verb ' + verb[0] +
                       ' found in ' + str(verb.root()) + '\n\nFound ' +
                       str(found))
                continue
            elif len(found) > 1:
                counts_by_function[i][3] += 1
                lex_verbs[verb_lemma][2][1] += 1
                verify('Found multiple ' + str(arg_types[i][0]) + 's of verb '
                       + verb[0] + ' in\n' + str(verb.root()) + '\n\nFound ' +
                       str(found))
                # If verify passes, then just use the leftmost non-null child
                newfound = []
                for j in range(len(found)):
                    if isinstance(found[j], ParentedTree):
                        newfound.append(found[j])
                        break
                found = newfound
            if len(found) == 1:
                # Switch to the head of the NP we found, if it's there (if it's
                # not, then the first is_noun will catch it).
                arg = find_head(found[0])
                if is_noun(arg):
                    if is_unmarked(arg):
                        # see if it's correct, and keep stats accordingly
                        if correct_tree[arg.treeposition()].label()[-1]\
                                == case_frame[i]:
                            # mark the noun (deep modification)
                            tree = mark(tree[arg.treeposition()],
                                        case_frame[i])
                            counts_by_function[i][0] += 1
                            # Count which verbs succeed most often.
                            lex_verbs[verb_lemma][0] += 1
                        else:
                            # Lexically specified case is not correct.
                            # First, verify it's just the case that's wrong.
                            assert correct_tree[
                                arg.treeposition()].label()[:-1] \
                                == arg.label()[:-1]
                            # Keep track of verbs that fail most often.
                            corr_case = correct_tree[
                                arg.treeposition()].label()[-1]
                            try:
                                counts_by_function[i][1] += 1
                                lex_verbs[verb_lemma][1][corr_case] += 1
                            except KeyError:
                                verify('Bad noun marked: ' + str(arg))
                    else:
                        # Found a noun, but it's already been marked
                        verify('We already marked the noun ' + str(found[0]) +
                               ', but it appears to be the ' + arg_types[i][0]
                               + ' of ' + str(verb) + ' in\n' +
                               str(verb.root()))
                else:
                    # Failed to find an N head of the NP we found
                    counts_by_function[i][2] += 1
                    lex_verbs[verb_lemma][2][0] += 1
                    verify('Found NP ' + str(found[0]) + ' in ' +
                           str(verb.root()) + 'is not a noun. (looking for ' +
                           arg_types[i][0] + ' of verb ' + verb[0])
            else:
                print('Error with number of', str(arg_types[i][0]) +
                      's found:', found)
                sys.exit(1)

    return tree

# #####################################################################
# #####################################################################
# ############################### Main ################################
# #####################################################################
# #####################################################################

# Control flow to choose which steps of which algorithms to test
baseline_steps = [False, False, False]
gfba_steps = [False, False, False, False, False]
sba_steps = [True, True, True, True, False]
safe_mode = False
print_errors = False

try:
    # CORPUS = open(sys.argv[1], encoding='utf-8')
    # CORPUS = open('testcorp.txt', encoding='utf-8')
    CORPUS = open('icepahc-v0.9/psd/2008.ofsi.nar-sag.psd', encoding='utf-8')
    # CORPUS = open('moderntexts.txt', encoding='utf-8')
    # CORPUS = open('alltexts.txt', encoding='utf-8')
except OSError:
    print('File not found.')
    sys.exit(1)

# Build the lexicon from the text file of verbs and prepositions
LEXICON = {}
lexfile = open('lexcasemarkers.txt', encoding='utf-8')
newline = lexfile.readline()
while newline:
    newline = newline.strip()
    if newline[0] != '#':
        LEXICON[newline[:newline.index(':')]] =\
            newline[newline.index(':') + 2:].replace(' ', '').split(',')
    newline = lexfile.readline()
del newline
lexfile.close()

corp_counts = {'N': 0, 'A': 0, 'D': 0, 'G': 0}
test_counts = {'N': 0, 'A': 0, 'D': 0, 'G': 0, '@': 0}

# Each sub-list is for the roles subject, indirect object, and direct object.
# The three elements of a sub-list represent the number of nouns of the given
# type that are marked correctly, marked incorrectly, left unmarked because
# no candidate was found and left unmarked because more than one candidate
# was found.
counts_by_function = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# A dictionary to keep track of which lexical verbs succeed and fail most
# often. Each key should be a string corresponding to a verb that lexically
# marks case. Each value should be a pair corresponding to the number of times
# that the verb marks an argument correctly and incorrectly.
lex_verbs = {}

# The scorecard will keep track of case marked by the algorithm versus case
# as marked in the corpus. The first letter is the corpus case; the second is
# the case marked by the algorithm. Thus, the total correct is the sum of NN,
# AA, DD, and GG.
scorecard = {'NN': 0, 'NA': 0, 'ND': 0, 'NG': 0,
             'AN': 0, 'AA': 0, 'AD': 0, 'AG': 0,
             'DN': 0, 'DA': 0, 'DD': 0, 'DG': 0,
             'GN': 0, 'GA': 0, 'GD': 0, 'GG': 0,
             'N@': 0, 'A@': 0, 'D@': 0, 'G@': 0}

newline = CORPUS.readline()

# Until the file is empty ...
while newline:
    current_tree = ''
    if newline == '\n':
        newline = CORPUS.readline()

    # ... read in a tree ...
    while newline and newline != '\n':
        current_tree += newline
        newline = CORPUS.readline()

    #####################################
    # ... and make a copy of the tree for later comparison. Count case
    # frequencies to update total count, and strip case from the test copy.
    # NOTE: I use the character @ for an unmarked case slot.
    #####################################
    corpus_tree = ParentedTree.fromstring(current_tree)
    corp_counts = count_case_freq(corpus_tree, corp_counts)
    current_tree = strip_case(ParentedTree.fromstring(current_tree))

    #######################################################
    # Now, update the case according to the given algorithm
    #######################################################
    # ## (1a) "Everything Nominative" (most frequent case) algorithm
    if baseline_steps[0]:
        current_tree = mark_all(current_tree, 'N')

    # ## (1b) "Random proportions" algorithm
    if baseline_steps[1]:
        current_tree = mark_random(current_tree, corp_counts)

    # ## (1c) "Total randomness" algorithm
    if baseline_steps[2]:
        current_tree = mark_random(current_tree)

    ##########
    # ## (2) Structure-Based Algorithm
    # Assign case based on an adaptation of the the structural algorithm
    # described by McFadden (2004)
    ##########

    # ## STEP 1: Lexically marked case
    # ## 1a: "quirky" verbs and prepositions
    if sba_steps[0]:
        # Being extra careful not to mess with what we're iterating over
        cp_tree = ParentedTree.fromstring(str(current_tree))
        for node in cp_tree.subtrees():
            if is_verb(node):
                try:
                    # extract the lemma of the verb from the tree node
                    quirky_verb = node[0][node[0].index('-') + 1:]
                    if quirky_verb in LEXICON:
                        new_tree = mark_args(node, LEXICON[quirky_verb][0],
                                             corpus_tree)
                        # If there are multiple case frames and the first
                        # didn't change anything, then try the second.
                        # NOTE: this code is only executed if the first has
                        # no effect. If the first has an effect, the second
                        # won't be tested, even if the second specifies a
                        # different argument from the first.
                        # Given my current lists, this is not a problem, but
                        # it would be ideal to allow testing of each argument
                        # for each frame given. For now, this relies on careful
                        # choice and/or modification of case frames.
                        if current_tree == new_tree and \
                           len(LEXICON[quirky_verb]) > 1:
                            new_tree = mark_args(node, LEXICON[quirky_verb][1],
                                                 corpus_tree)
                        # Make the changes from mark_args
                        current_tree = new_tree

                except ValueError:
                    verify('Can\'t find dash char to find lemma of verb '
                           + node[0] + ' in tree\n' + str(node.root()))
        del cp_tree

    # ## For efficiency, keep track of all unmarked nouns instead of searching
    # ## the whole tree at each of the following steps.
    unmarked_nouns = []
    for node in current_tree.subtrees():
        if is_noun(node) and is_unmarked(node):
            # add both the node and the base position of its maximal
            # projection (find_base_pos is benign if the node wasn't moved
            # (or A'-moved, if you want to exclude A-movement)
            unmarked_nouns.append([node.treeposition(),
                                   find_base_pos(
                find_max_proj(node)).treeposition()])

    # ## STEP 1B: "Applicative" datives (indirect objects), genitive
    # possessors, and dative prepositional objects
    if sba_steps[1]:
        for pos in unmarked_nouns[:]:
                if current_tree[pos[1]].parent().label()[:2] == 'PP' \
                        or current_tree[pos[1]].parent().label()[:3] == 'WPP':
                    # if it's a preposition, mark the case specified in the
                    # lexicon (if that exists)
                    unmarked_nouns.remove(pos)
                    try:
                        for c in current_tree[pos[1]].parent():
                            if c.label() == 'P':
                                for frame in \
                                        LEXICON[c[0][c[0].index('-') + 1:]]:
                                    if frame[2] \
                                            == corpus_tree[pos[0]].label()[-1]:
                                        mark(current_tree[pos[0]], frame[2])
                                        break
                    except (ValueError, KeyError):
                        current_tree = mark(current_tree[pos[0]], 'D')
                    # If that failed, just mark dative as a default inside PP.
                    if is_unmarked(current_tree[pos[0]]):
                        current_tree = mark(current_tree[pos[0]], 'D')

                elif current_tree[pos[1]].label()[:6] in ['NP-OB2', 'NP-OB3'] \
                        or current_tree[pos[1]].label()[:7] in ['WNP-OB2',
                                                                'WNP-OB3']:
                    unmarked_nouns.remove(pos)
                    current_tree = mark(current_tree[pos[0]], 'D')
                elif current_tree[pos[1]].label()[:6] == 'NP-POS' \
                        or current_tree[pos[1]].label()[:7] == 'WNP-POS':
                    unmarked_nouns.remove(pos)
                    current_tree = mark(current_tree[pos[0]], 'G')

    # ## STEP 2: Dependent case
    if sba_steps[2]:
        to_be_marked_acc = []
        for pos in unmarked_nouns:
            for pos2 in unmarked_nouns:
                n1 = current_tree[pos[1]]
                n2 = current_tree[pos2[1]]
                if c_commands(n1, n2) and precedes(n1, n2) and \
                        same_domain(n1, n2):
                    # avoid duplicates
                    if pos2 not in to_be_marked_acc:
                        to_be_marked_acc.append(pos2)
        for t in to_be_marked_acc:
            unmarked_nouns.remove(t)
            current_tree = mark(current_tree[t[0]], 'A')

    # ## STEP 3: Unmarked case
    if sba_steps[3]:
        for pos in unmarked_nouns[:]:
            par = current_tree[pos[1]].parent()
            while par is not None:
                if par.label()[:2] == 'CP' or par.label()[:6] == 'IP-MAT':
                    unmarked_nouns.remove(pos)
                    current_tree = mark(current_tree[pos[0]], 'N')
                    break
                elif par.label()[:2] in ['NP', 'PP'] \
                        or par.label()[:3] in ['WNP', 'WPP']:
                    break
                else:
                    par = par.parent()

    # ## STEP 4: Default
    if sba_steps[4]:
        for pos in unmarked_nouns[:]:
            current_tree = mark(current_tree[pos[0]], 'N')

    #######################################################
    # ## (3) Grammatical-function-based algorithm
    # ##     Using the NP-<func> markings given in the corpus,
    # ##     make the following case assignments:
    # ##     NOM to subjects             NP-SBJ
    # ##     ACC to direct objects       NP-OB1
    # ##     DAT to indirect objects     NP-OB2 and NP-OB3 & PP objects
    # ##     GEN to possessives          NP-POS
    #######################################################

    if gfba_steps[0] or gfba_steps[1] or gfba_steps[2] or gfba_steps[3] or \
       gfba_steps[4]:
        # Being extra careful not to mess with what we're iterating over
        cp_tree = ParentedTree.fromstring(str(current_tree))
        for node in cp_tree.subtrees():
            if is_noun(node) and is_unmarked(node):
                if gfba_steps[0] and find_func(node, 'SBJ'):
                    current_tree = mark(current_tree[node.treeposition()], 'N')
                elif gfba_steps[1] and find_func(node, 'OB1'):
                    current_tree = mark(current_tree[node.treeposition()], 'A')
                elif gfba_steps[2] and \
                        (find_func(node, 'OB2') or find_func(node, 'OB3')):
                    current_tree = mark(current_tree[node.treeposition()], 'D')
                elif gfba_steps[3] and find_func(node, 'PPOBJ'):
                    current_tree = mark(current_tree[node.treeposition()], 'D')
                elif gfba_steps[4] and find_func(node, 'POS'):
                    current_tree = mark(current_tree[node.treeposition()], 'G')
        del cp_tree

    # ####################################
    # ... and match the tree's cases against the corpus version and update
    # the total scores.
    # ####################################
    test_counts = count_case_freq(current_tree, test_counts)
    scorecard = score_tree(corpus_tree, current_tree, scorecard)

    if print_errors and current_tree != corpus_tree:
        print(current_tree, '\n\n')
    #####################################

# Finally, print statistics from the tree...
print_counts(corp_counts, 'corpus')
print()
print_counts(test_counts, 'trees marked by algorithm')
print()
# ... statistics on the lexically-marked words by function
print('Number of attempts to mark arguments of quirky verbs, formatted as')
print('[marked correctly, found but incorrect in lexicon, found none,',
      'found too many, total attempts]')
for i in range(len(counts_by_function)):
    counts_by_function[i] += [sum(counts_by_function[i])]
print('\t   Subjects:', counts_by_function[0])
print('\tInd Objects:', counts_by_function[1])
print('\tDir Objects:', counts_by_function[2])
print('\t     Totals:', [sum(counts_by_function[i][j]
                             for i in range(len(counts_by_function)))
                         for j in range(len(counts_by_function[0]))])
# ... and the scorecard (incl_unmarked includes unmarked nouns in calculation
# of precision, recall, and f-score).
pp_score(scorecard, incl_unmarked=False)

# Print most successful and least successful lexical verbs.
# Use the following key to sort by most frequently marking wrong case
# key=lambda x: sum(lex_verbs[x][1].values())
# Use this key to sort by most frequently failing to assign case
# key=lambda x: sum(lex_verbs[x][2])
# Set the reverse key to True/False to see least/most successful, resp.
print('Least successful quirky verbs:')
for vb in sorted(lex_verbs.keys(), key=lambda x: sum(lex_verbs[x][1].values()),
                 reverse=True):
    print(vb, lex_verbs[vb],
          '"Wrong" on list:', sum(lex_verbs[vb][1].values()),
          '       Unmarked:', sum(lex_verbs[vb][2]))
print('              Total:', sum(sum(lex_verbs[v][1].values())
                                  for v in lex_verbs.keys()))
print('Total unmarked:', sum(sum(lex_verbs[v][2]) for v in lex_verbs.keys()))
CORPUS.close()
