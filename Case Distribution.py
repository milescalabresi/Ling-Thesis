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
import os
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


def c_commands(a, b):
    """
    :param a: any node in a tree
    :param b: any node in the same tree
    :return: Boolean value
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


def is_noun(label):
    """
    A Predicate to tell whether a given node in the tree is a noun
    and thus able to be marked with case.
    :param label: either a node or its label
    :return: Boolean
    """
    if type(label) != str:
        label = label.label()
    # the @ symbol is my placeholder character for "unmarked"
    if re.match('(N(PR)?S?|W?PRO)-[NADG@]', label):
        return True
    else:
        return False


def is_verb(st):
    """
    A Predicate to tell whether a given node in a tree is a verb.
    :param st: a node in some tree
    :return: Boolean corresponding to whether the node is a verb
    """
    assert type(st) != str
    return type(st[0]) == str and st.label()[:2] == 'VB'


def is_unmarked(word):
    """
    Predicate to determine whether a given noun (or pronoun) is currently
    unmarked for case
    :rtype : Boolean
    :param word: any node (presumably a noun head) in a atree
    :return: Boolean value corresponding to if word is not case-marked
    """
    if type(word) != str:
        word = word.label()
    if is_noun(word):
        return word[-1] == '@'
    else:
        if verify('Error: The node ' + word +
                  ' inputted is not a noun; cannot check case marking.'):
            return False
        else:
            sys.exit(2)


def find_base_pos(word):
    """
    Find the base position of a moved constituent
    :param word: a node in the tree
    :return: a different node corresponding to the base position of word
    """
    num = word.label()[-2:]
    assert re.match('-\d', num)
    found = []
    for st in word.root():
        if st.label() == '*ICH*' + num:
            found += st
    if len(found) == 1:
        return found[0]
    else:
        print('Error finding unique base position for', word.label(),
              'in tree', word.root(), 'Found', found)
        if verify():
            return word
        else:
            sys.exit(1)


def find_surf_pos(word):
    """
    Find the base position of a moved constituent
    :param word: a node in the tree
    :return: a different node corresponding to the surface position of word
    """
    num = word[0][-1]
    assert re.match('\d', num)
    found = []
    for st in word.root().subtrees():
        if st.label()[-2:] == '-' + num:
            found.append(st)
    if len(found) == 1:
        return found[0]
    else:
        print('Error finding unique surface position for', word,
              'in tree', word.root(), 'Found', found)
        if verify():
            return word
        else:
            sys.exit(1)


def mark(word, case):
    """
    Given a node in a tree and a case string, mark that node (which should be
    a noun) as the given case.
    :param word: a node in the tree
    :param case: a string describing a given case
    :return:
    """
    if not is_noun(word):
        if verify(str(word) + ' doesn\'t look like not a noun.'):
            return word
    if case != '@':
        if not is_unmarked(word):
            return word
    word.set_label(word.label()[:-1] + case)
    return word


def mark_all(tree, case):
    """
    a function to mark all noun heads in a given tree the same case
    :param tree: any tree
    :param case: any case marking (as a string)
    :return: the same tree, but with all nouns marked in the given case
    """
    for st in tree.subtrees():
        if is_noun(st):
            tree[st.treeposition()] = mark(st, case)
    return tree


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
    :param tree:
    :param counter:
    :return:
    """
    for word in tree.pos():
        if is_noun(word[1]):
            try:
                counter[word[1][-1]] += 1
            except KeyError:
                print('Error: bad case ' + word[1][-1] + ' on ' + str(word))
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
    corp_nodes = corpus.pos()
    test_nodes = test.pos()
    assert len(corp_nodes) == len(test_nodes)

    pairs = [(corp_nodes[i][1], test_nodes[i][1])
             for i in range(len(test_nodes)) if is_noun(test_nodes[i][1])]

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
            print('\t', (key, str(round(100.0 * counts[key] /
                  sum(counts.values()), 3)) + '%'))
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
    if right == marked_wrong == 0:
        pct = 0.000
    else:
        if incl_unmarked:
            pct = round(100.0 * right / (unmarked + marked_wrong + right), 3)
        else:
            pct = round(100.0 * right / (marked_wrong + right), 3)

    print('Total marked correctly:', str(right), '\t(' + str(pct) + '%)')
    print('Total left unmarked:', unmarked)
    print('Total marked incorrectly:', marked_wrong)
    print('Total wrong:', unmarked + marked_wrong)

    print('\nSTATISTICS BY INDIVIDUAL CASE')
    for case in 'NADG':
        cases = 'NADG'
        if incl_unmarked:
            cases += '@'
        relevant = float(sum([card[case + case2] for case2 in cases]))
        selected = float(sum([card[case2 + case] for case2 in 'NADG']))
        correct = float(card[case + case])
        precision = correct/selected
        recall = correct/relevant
        print('Case:', case)
        print('\tPrecision:', str(round(100*precision, 3)) + '%')
        print('\t   Recall:', str(round(100*recall, 3)) + '%')
        print('\t  F-score:', str(round(100*f_score(precision, recall), 3))
              + '%')


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
    return (1 + beta * beta) * float(precision * recall) / \
        float(beta * beta * precision + recall)


def verify(msg=''):
    """
    Verify that it's okay to continue or quit; used to quit if a possible
    error arises.
    :param msg: an optional text message to explain what needs to be verified
    :return: nothing -- either continue or quit
    """
    return True  # Uncomment to see what happens!
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
    for st in tree.subtrees():
        if is_noun(st) and is_unmarked(st):
            tree[st.treeposition()] = mark(st, choose_rand_weighted(freqs))
    return tree


def find_head(np):
    """
    Search descendants of a given noun phrase layer for the head of the NP.
    :param np: a NP projection in a tree
    :return: the N head of that NP (in that same tree)
    """
    if is_noun(np.label()):
        return np
    if np.label()[:2] != 'NP' and np.label()[:3] != 'WNP':
        print('Bad noun phrase in find_head function:', np)
        sys.exit(1)
    for child in np:
        if is_noun(child):
            return child
    verify('No N head child of NP ' + str(np))
    return np


def find_func(n_head, func):
    """
    Search ancestors of a given noun head for a function-marked NP layer, up to
    CP layers
    :param n_head: a noun head node in any tree
    :param func: a string describing the grammatical function of the sought NP
    :return: a node in the same tree corresponding to NP with the given
    function that contains the given N head, or None if none found
    """
    while (n_head.parent() is not None) and \
            (n_head.label()[:2] != 'CP') and \
            (n_head.label()[:6] != 'NP-SBJ') and \
            (n_head.label()[:5] != 'NP-OB') and \
            (n_head.label()[:6] != 'NP-POS') and \
            (n_head.label()[:2] != 'PP'):
        n_head = n_head.parent()
        if n_head.label() == 'NP-' + func:
            return n_head
        elif n_head.label()[:2] == 'PP' and func == 'PPOBJ':
            return True
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


def crosses(st, ancestor, layer):
    """
    determine whether one crosses a layer of type layer while going from
    the node st to the node ancestor (or vice versa)
    :param st: any node in a tree
    :param ancestor: an ancestor of st, including possibly st itself
    :param layer: the type of node to test for, e.g. CP
    :return:
    """
    assert st.root() == ancestor.root()
    assert ns_dominates(ancestor, st)
    if st == ancestor:
        return False
    st = st.parent()
    while st != ancestor:
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
    return not (crosses(a, lca, 'CP') or crosses(b, lca, 'CP'))


def mark_args(verb, case_frame, correct_tree):
    """
    Take a lexical item (verb, and perhaps later preposition) and find the
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
        print('Error with lexical case frame', '"' + case_frame + '"')
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
                 ['direct object', 'OB1', 'OB1'],
                 ['indirect object', 'OB2', 'OB3']]
    # For i = 2 (indirect objects), we need to check if it might OB3 or OB2
    # For i < 2, we just check the first condition again (hence the redundant
    # third item in the list.

    for i in range(len(case_frame)):
        if case_frame[i] != '-':
            found = []
            if i == 0:  # looking for subjects
                cc_cond = lambda x, y: c_commands(x, y)
            else:  # looking for some kind of object
                cc_cond = lambda x, y: c_commands(y, x)

            for st in tree.subtrees():
                if (st.label()[:6] == 'NP-' + str(arg_types[i][1]) or
                   st.label()[:6] == 'NP-' + str(arg_types[i][2])) and \
                   cc_cond(st, verb) and same_domain(st, verb):
                    if st[0][-7:-2] == '*ICH*' or st[0][-5:-2] == '*T*':
                        st = find_surf_pos(st)
                    found.append(st)

            # Now keep statistics of whether each verb's arguments
            # were found and marked correctly or incorrectly.
            verb_lemma = verb[0][verb[0].index('-') + 1:] + ':' \
                + str(i+1) + '-' + case_frame[i]
            if verb_lemma not in lex_verbs:
                lex_verbs[verb_lemma] = [0, {'N': 0, 'A': 0, 'D': 0, 'G': 0},
                                         [0, 0]]

            if len(found) == 0:
                counts_by_function[i][2] += 1
                lex_verbs[verb_lemma][2][0] += 1
                verify('No subject of verb ' + verb[0] + ' found in ' +
                       str(verb.root()) + '\n\nFound ' + str(found))
            elif len(found) > 1:
                counts_by_function[i][3] += 1
                lex_verbs[verb_lemma][2][1] += 1
                verify('Found multiple ' + str(arg_types[i][0]) + 's of verb '
                       + verb[0] + ' in\n' + str(verb.root()) + '\n\nFound ' +
                       str(found))
            elif len(found) == 1:
                found[0] = mark(find_head(found[0]), case_frame[i])
                tree[found[0].treeposition()] = found[0]  # deep modification
                if correct_tree[found[0].treeposition()] == found[0]:
                    counts_by_function[i][0] += 1
                    # Count which verbs succeed most often.
                    lex_verbs[verb_lemma][0] += 1
                else:
                    counts_by_function[i][1] += 1
                    # Verbs that fail most often
                    corr_case = find_head(
                        correct_tree[found[0].treeposition()]).label()[-1]
                    try:
                        lex_verbs[verb_lemma][1][corr_case] += 1
                    except KeyError:
                        verify('Bad noun marked: ' + str(found[0]))
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


try:
    # CORPUS = open(sys.argv[1], encoding='utf-8')
    # CORPUS = open('testcorp.txt', encoding='utf-8')
    CORPUS = open('icepahc-v0.9/psd/2008.ofsi.nar-sag.psd', encoding='utf-8')
    # CORPUS = open('Modern IcePaHC Files.txt', encoding='utf-8')
except OSError:
    print('File not found.')
    sys.exit(1)

# Build the lexicon from the csv file of verbs and prepositions
LEXICON = {}
lexfile = open('lexcasemarkers.txt', encoding='utf-8')
newline = lexfile.readline()
while newline:
    LEXICON[newline[:newline.index(':')]] = newline[newline.index(': ') + 2:-1]
    newline = lexfile.readline()
del newline
lexfile.close()


corp_counts = {'N': 0, 'A': 0, 'D': 0, 'G': 0}
test_counts = {'N': 0, 'A': 0, 'D': 0, 'G': 0, '@': 0}

# Each sub-list is for the roles subject, object, direct object
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
    # current_tree = mark_all(current_tree, 'N')

    # ## (1b) "Random proportions" algorithm
    # current_tree = mark_random(current_tree, corp_counts)

    # ## (1c) "Total randomness" algorithm
    # current_tree = mark_random(current_tree)

    #######################################################
    # ## (2) "Naive" grammatical-function-based algorithm
    # ##     Using the NP-<func> markings given in the corpus,
    # ##     make the following case assignments:
    # ##     NOM to subjects             NP-SBJ
    # ##     ACC to direct objects       NP-OB1
    # ##     DAT to indirect objects     NP-OB2 and NP-OB3
    # ##     GEN to possessives          NP-POS
    #######################################################

    if False:
        for node in current_tree.subtrees():
            if is_noun(node) and is_unmarked(node):
                if find_func(node, 'SBJ'):
                    node = mark(node, 'N')
                elif find_func(node, 'OB1'):
                    node = mark(node, 'A')
                elif find_func(node, 'OB2') or find_func(node, 'OB3'):
                    node = mark(node, 'D')
                elif find_func(node, 'POS'):
                    node = mark(node, 'G')
                elif find_func(node, 'PPOBJ'):
                    node = mark(node, 'D')

    ##########
    # ## (3) Structure-Based Algorithm
    ##########

    # ## STEP 1: Lexically marked case
    for node in current_tree.subtrees():
        if is_verb(node):
            try:
                quirky_verb = node[0][node[0].index('-') + 1:]
                if quirky_verb in LEXICON:
                    current_tree = mark_args(node, LEXICON[quirky_verb],
                                             corpus_tree)
            except ValueError:
                verify('Can\'t find dash char to find lemma of verb '
                       + node[0] + ' in tree\n' + str(node.root()))

    # ## STEP 2: Dependent case
    # for node in current_tree.subtrees():
    #     if is_noun(node) and is_unmarked(node):
    #         for node2 in current_tree.subtrees():
    #             if is_noun(node2) and is_unmarked(node) and node != node2 \
    #                and c_commands(node, node2) and same_domain(node, node2):
    #                     node2 = mark(node2, 'A')

    # ## STEP 3: Unmarked case
    # for node in current_tree.subtrees():
    #     if is_noun(node) and is_unmarked(node):
    #         par = node.parent()
    #         while par is not None:
    #             if par.label()[:2] == 'CP':
    #                 node = mark(node, 'N')
    #                 break
    #             elif par.label()[:2] == 'PP':
    #                 node = mark(node, 'D')
    #                 break
    #             #elif par.label()[:2] == 'NP':
    #             #    node = mark(node, 'G')
    #             #    break
    #             else:
    #                 par = par.parent()
    #         del par

    # ## STEP 4: Default
    # current_tree = mark_all(current_tree, 'N')

    # ####################################
    # ... and match the tree's cases against the corpus version and update
    # the total scores.
    # ####################################
    test_counts = count_case_freq(current_tree, test_counts)
    scorecard = score_tree(corpus_tree, current_tree, scorecard)
    #####################################

# Finally, print statistics from the tree...
print_counts(corp_counts, 'corpus')
print()
print_counts(test_counts, 'trees marked by algorithm')
print()
# ... statistics on the lexically-marked words by function
print('Number of attempts to mark arguments of quirky verbs, formatted as')
print('[marked correctly, marked incorrectly, found none, found too many]')
print('\t        Subjects:', counts_by_function[0])
print('\t  Direct Objects:', counts_by_function[1])
print('\tIndirect Objects:', counts_by_function[2])
# ... and the scorecard.
pp_score(scorecard, incl_unmarked=False)

# Print most successful and least successful lexical verbs.
for vb in sorted(lex_verbs.keys(), key=lambda x: sum(lex_verbs[x][1].values()),
                 reverse=True):
        print(vb, lex_verbs[vb], 'Total wrong:',
              sum(lex_verbs[vb][1].values()))
CORPUS.close()