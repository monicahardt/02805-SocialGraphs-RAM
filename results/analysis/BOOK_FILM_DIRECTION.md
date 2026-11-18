# Research direction: how language changes in adaptation

Primary question: **How does adapting the Harry Potter books into films change the language used to portray characters, particularly villains?**

The unit of comparison is a matched character and story installment. Compare normalized rates and visible raw support. For installment seven, combine the two films before treating them as the counterpart to the book. Use Hagrid and other protagonists as comparison cases where they help distinguish general adaptation changes from villain-specific characterization.

## A small set of connected visual questions

| Visual question | Course methods | Display | What the reader can learn |
| --- | --- | --- | --- |
| Which words survive or become concentrated? | Counts, normalized rates, TF-IDF | A shared-word frequency comparison with book-only and film-only side panels; select a word to see quotations | Separate repeated signature vocabulary, omitted material, new wording and names tied to particular scenes. A static Umbridge prototype is in `results/comparison/umbridge_phoenix/`. |
| Are differences explained by shortening alone? | TF-IDF and an explicit random-removal benchmark | A distribution of equal-length random cuts with the actual film result marked | Compare exact-word preservation with a transparent null model. The first case uses TF-IDF-weighted vocabulary coverage; this is a custom summary of the existing weights, not a new NLP model or a measure of preserved meaning. |
| Which themes are preserved or redistributed? | Jointly fitted LDA | Paired topic profiles per character and matched installment, with top words and source passages beside a selected topic | Compare topic proportions while keeping the reference model and story segment fixed. Evaluate seeds/topic counts before interpreting small changes. |
| Does the same word acquire different surroundings? | Word/context counts and PPMI | Book and film context graphs with fixed node positions; show shared, book-associated and film-associated edges with support counts | Explore how the contexts of blood, power, death or love differ. Recompute PPMI separately by medium with a shared vocabulary; pooled PPMI alone cannot answer this. |
| Which spell references remain, and who uses them? | Phrase occurrence counts, contextual inspection, PPMI | A paired character-by-spell heatmap with matching installments and quotation drill-down | Distinguish mentions, descriptions and casting. Show counts and source coverage. This remains a future step. |

## What makes the presentation meaningful

Every view should compare books with films directly, expose the denominator and support, and let the reader return to the relevant text. Use a consistent visual identity for books and films. Show absent terms separately from terms outside a displayed top-k list.

Interaction should answer a research question: choosing a character, fixing an installment or opening evidence. Aesthetic novelty should help reveal preservation, compression or altered context.

LDA topic-mixture changes do not establish that particular sentences or themes were transferred between topics. A Sankey diagram of topic-to-topic flow would require an explicit, validated alignment of book passages with film dialogue; aggregate topic proportions alone do not supply those flows.

Start with the Umbridge/Order of the Phoenix case. Then test whether the proposed pattern survives broader vocabulary definitions, other characters, source verification, equal-length comparisons and modeling choices. Keep this as a sequence of small, reviewable steps.
