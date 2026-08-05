# rag
explore the rag pipelines in LangChain and in LlamaIndex


- Initial results
    - **Vector db backed retriever**
    - 📊  Mean Scores
        - context_precision    0.413333
        - context_recall       0.385417
        - faithfulness         0.673333
        - answer_relevancy     0.639957

    - **Parentdocument retriever**
    - 📊  Mean Scores
        - context_precision    0.783333
        - context_recall       0.815942
        - faithfulness         0.931677
        - answer_relevancy     0.807388
    
    - 📊  Median Scores
        - context_precision    1.000000
        - context_recall       1.000000
        - faithfulness         1.000000
        - answer_relevancy     0.899437

    - insights
        - The baseline vector search succeeded on simple financial queries, but failed catastrophically on complex narrative or multi-part queries (driving down the mean while keeping medians at 1.00).
        - Outright Retrieval Collapses (Zero Recall & Precision)
            - Row 3: "How many people worked at Apple as of the end of FY2025..." (Headcount)
            - Row 12: "Apple's FY2023 had an extra week compared to FY2024..." (Fiscal Calendar)
            - Pure dense vector embeddings prioritize semantic topic matching over specific non-numeric corporate disclosures. Standard chunking severed the headcount and fiscal-calendar notes from the main financial tables, leading to zero retrieved context.
        - Partial Context Fragmentation & Context Loss
            - Row 8 (Accounting Footnote): Context Precision dropped to 0.50, Context Recall to 0.50. Dense vector search retrieved the table numbers but missed Footnote 1 at the bottom of the page.
            - Row 13 (Risk Factor Concentration): Precision dropped to 0.33, Recall to 0.67. The retriever pulled the numerical table (iPhone sales ratio) but completely missed the narrative Risk Factors section located elsewhere in the 10-K.
            - Row 21 (Effective Tax Rate & State Aid Decision): Precision dropped to 0.0. The retriever pulled income tax tables but missed the explanatory text detailing the European Commission State Aid decision.