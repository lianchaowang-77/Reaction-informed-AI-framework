Supplementary Data for "Reaction-informed AI design of acid-responsive hydrolyzable molecules for marine antifouling"



Overview
This package provides the data chain needed to inspect and reconstruct the main data-driven conclusions of the study: experimental hydrolysis data, initial and selected reaction-descriptor (RD) matrices, DFT-surrogate training data, high-throughput screening results, substituent enrichment and mutation records, targeted-search results, and the final beyond-limit candidate list.

Files

1. Experimental\_dataset\_184.csv: standardized experimental structures and hydrolysis-rate values for 184 molecules.
2. Initial\_RD\_matrix\_184x408.csv: 408 initial RD features for the 184 experimental molecules, plus identifiers and outcomes.
3. Final\_RD\_matrix\_184x51.csv: 51 selected RD features used for model training, plus identifiers and outcomes.
4. DFT\_surrogate\_training\_1500.csv: structures, eight DFT-derived target properties, and target-specific data splits for the eight single-task surrogate models.
5. Initial\_virtual\_space\_465504.csv: predictions and screening metadata for 465,504 candidates.
6. High\_value\_and\_initial\_38\_candidates.csv: the 12,543 candidates with Predicted\_log\_kH > 4.0, including 38 candidates with Predicted\_log\_kH > 6.1.
7. Substituent\_pool\_and\_mutations.csv: 55 enriched and 60 mutated site-specific substituent records used to define the targeted-search space.
8. Targeted\_search\_225\_candidates.csv: 225 unique candidates with Predicted\_log\_kH > 6.1 found by targeted search.
9. All\_263\_beyond\_limit\_candidates.csv: the combined set of 38 HTS and 225 targeted-search candidates.



Identifiers and molecular structures
Molecule\_ID, Candidate\_ID, and Substituent\_ID are unique within their respective entity types and remain consistent across files. SMILES is a canonical, non-isomeric representation generated with RDKit and is used for structural deduplication. Substituent attachment points are represented by an asterisk (\*).



Hydrolysis-rate definitions
k\_H denotes the second-order acid-catalyzed hydrolysis rate constant. Experimental\_log\_kH and Predicted\_log\_kH are base-10 logarithms.



Thresholds
High\_value\_flag is TRUE when Predicted\_log\_kH > 4.0. Beyond\_limit\_flag is TRUE when Predicted\_log\_kH > 6.1. Values equal to a threshold are not included in the corresponding category.



SA score and logP
SA\_score denotes the unitless synthetic accessibility score, with lower values indicating molecules that are generally easier to synthesize.
logP denotes the predicted octanol/water partition coefficient calculated using the RDKit Crippen method, with higher values indicating greater molecular lipophilicity.



HTS denotes high-throughput screening.

