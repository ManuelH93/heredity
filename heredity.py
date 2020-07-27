import csv
import itertools
import sys
import pandas as pd
import numpy as np

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Create dataframe that stores name, genes, parent genes and trait
    # for each person.
    names = [name for name in set(people)]

    genes = []
    for name in set(people):
        if name in one_gene:
            genes.append(1)
        elif name in two_genes:
            genes.append(2)
        else:
            genes.append(0)

    mother_genes = []
    for name in set(people):
        mother = people[name]['mother']
        if mother == None:
            mother_genes.append(np.nan)
        elif mother in one_gene:
            mother_genes.append(1)
        elif mother in two_genes:
            mother_genes.append(2)
        else:
            mother_genes.append(0)

    father_genes = []
    for name in set(people):
        father = people[name]['father']
        if father == None:
            father_genes.append(np.nan)
        elif father in one_gene:
            father_genes.append(1)
        elif father in two_genes:
            father_genes.append(2)
        else:
            father_genes.append(0)

    trait = []
    for name in set(people):
        if name in have_trait:
            trait.append(True)
        else: trait.append(False)

    listofkeys = ('name', 'genes', 'mother_genes', 'father_genes', 'trait')
    listofvalues = (names, genes, mother_genes, father_genes, trait)
    dictionary=dict(zip(listofkeys,listofvalues))
    df = pd.DataFrame(dictionary)
    df.set_index('name', inplace=True)

    # Determine how gene is distributed across parents,
    # if parents are known
    df['scenario'] = df.apply(scenarios, axis=1)

    # Calculate probability of having number of genes
    # specified in input tp the functin             
    df['prob_gene'] = df.apply(prob_gene, args=(one_gene, two_genes), axis=1)

    # Depending on 'have_trait', the below calculates the probability
    # that an individual either has or doesn't have the trait.
    df['prob_trait'] = df.apply (lambda row: PROBS['trait'][row.genes][row.trait], axis=1)

    # Combined probability of genes and trait
    df['prob_combined'] = df.apply (lambda row: row.prob_gene * row.prob_trait, axis=1)

    # Joint probability
    result = df['prob_combined'].product(axis = 0)
    return result


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name in set(probabilities):
        # Update gene
        if name not in one_gene and name not in two_genes:
            probabilities[name]['gene'][0] += p
        if name in one_gene:
            probabilities[name]['gene'][1] += p
        if name in two_genes:
            probabilities[name]['gene'][2] += p
        # Update trait
        if name not in have_trait:
            probabilities[name]['trait'][False] += p
        if name in have_trait:
            probabilities[name]['trait'][True] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in set(probabilities):
        # Update gene
        total_gene = 0
        for i in range(3):
            total_gene += probabilities[name]['gene'][i]
        for j in range(3):
            probabilities[name]['gene'][j] = (probabilities[name]['gene'][j]) / total_gene
        # Update trait
        total_trait = 0
        for k in (True, False):
            total_trait += probabilities[name]['trait'][k]
        for l in (True, False):
            probabilities[name]['trait'][l] = (probabilities[name]['trait'][l]) / total_trait


def scenarios(row):
    """
    Determine and return scenario.

    When calculating the probability that a child has a certain
    amount of genes, we need to first determine how the gene
    is distributed across the parents. There are six possible
    scenarios which are relevant for our calculation.
    """
    # No parent has gene
    if (row['mother_genes'] == 0) and row['father_genes'] == 0:
        return 'p_00'
    # Both parents have one gene
    if (row['mother_genes'] == 1) and row['father_genes'] == 1:
        return 'p_11'
    # Both parents have two genes
    if (row['mother_genes'] == 2) and row['father_genes'] == 2:
        return 'p_22'
    # One parent has zero genes, one parent has one gene 
    if (row['mother_genes'] == 0) and row['father_genes'] == 1:
        return 'p_01'
    if (row['mother_genes'] == 1) and row['father_genes'] == 0:
        return 'p_01'
    # One parent has zero genes, one parent has two genes 
    if (row['mother_genes'] == 0) and row['father_genes'] == 2:
        return 'p_02'
    if (row['mother_genes'] == 2) and row['father_genes'] == 0:
        return 'p_02'
    # One parent has one gene, one parent has two genes 
    if (row['mother_genes'] == 1) and row['father_genes'] == 2:
        return 'p_12'
    if (row['mother_genes'] == 2) and row['father_genes'] == 1:
        return 'p_12'


def con_prob_0(row):
    """
    Returns probability of having no gene,
    given different possible distributions of
    gene for parents.
    """
    if row['scenario'] == 'p_00':
        return 0.99 * 0.99
    if row['scenario'] == 'p_11':
        return 0.5 * 0.5
    if row['scenario'] == 'p_22':
        return 0.01 * 0.01
    if row['scenario'] == 'p_01':
        return 0.50 * 0.99
    if row['scenario'] == 'p_02':
        return 0.99 * 0.01
    if row['scenario'] == 'p_12':
        return 0.50 * 0.01


def con_prob_1(row):
    """
    Returns probability of having one gene,
    given different possible distributions of
    gene for parents.
    """
    if row['scenario'] == 'p_00':
        return 0.01 * 0.99 + 0.01 * 0.99
    if row['scenario'] == 'p_11':
        return 0.50 * 0.50 + 0.50 * 0.50
    if row['scenario'] == 'p_22':
        return 0.99 * 0.01 + 0.01 * 0.99
    if row['scenario'] == 'p_01':
        return 0.50 * 0.99 + 0.01 * 0.50
    if row['scenario'] == 'p_02':
        return 0.99 * 0.99 + 0.01 * 0.01
    if row['scenario'] == 'p_12':
        return 0.99 * 0.50 + 0.01 * 0.50


def con_prob_2(row):
    """
    Returns probability of having two genes,
    given different possible distributions of
    gene for parents.
    """
    if row['scenario'] == 'p_00':
        return 0.01 * 0.01
    if row['scenario'] == 'p_11':
        return 0.5 * 0.5
    if row['scenario'] == 'p_22':
        return 0.99 * 0.99
    if row['scenario'] == 'p_01':
        return 0.50 * 0.01
    if row['scenario'] == 'p_02':
        return 0.99 * 0.01
    if row['scenario'] == 'p_12':
        return 0.50 * 0.99


def prob_gene(row, one_gene, two_genes):
    """
    Returns probability for having a
    specific number of genes.
    """
    # Unconditional probability if parents unknown
    if row['scenario'] == None:
        return PROBS['gene'][row.genes]
    # Conditional probability if parents known
    else:
        if row.name in one_gene:
            return con_prob_1(row)
        elif row.name in two_genes:
            return con_prob_2(row)
        else:
            return con_prob_0(row)


if __name__ == "__main__":
    main()
