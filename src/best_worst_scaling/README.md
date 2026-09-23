# Best-Worst Scaling

Scripts by Svetlana Kiritchenko, Peter Turney and Saif Mohammad (National Research Council Canada). Their terms of use are in `readme.txt`; if you use the scripts, please cite their papers listed there. The originals and example files are on [Saif Mohammad's page](http://www.saifmohammad.com/WebPages/BestWorst.html).

Two settings differ from the originals, matching how we collected our data:

- `generate-BWS-tuples.pl` creates 2N 4-tuples for N items (`$factor = 2`).
- `get-scores-from-BWS-annotations-counting.pl` rescales the scores from -1 to 1 onto 1 to 5 (Equation 1 in the paper).

## Requirements

Perl with three modules:

```
cpan Text::CSV Statistics::Basic Statistics::RankCorrelation
```

## Usage

**1. Generate tuples** from a file with one item per line:

```
perl generate-BWS-tuples.pl items.txt
```

This writes `items.txt.tuples`, one tuple per line, items separated by tabs.

**2. Collect annotations** as a CSV with this header. Each row is one annotator's decision for one tuple:

```
Item1,Item2,Item3,Item4,BestItem,WorstItem
hold office,write thing,throw ball,undermine principle,throw ball,undermine principle
```

`BestItem` is the most concrete item, `WorstItem` the most abstract.

**3. Compute scores:**

```
perl get-scores-from-BWS-annotations-counting.pl annotations.csv > scores.tsv
```

Each output line is an item and its score, from 1 (abstract) to 5 (concrete).

**4. Check reliability** with split-half reliability over 100 random splits:

```
perl SHR-BWS.pl annotations.csv
```
