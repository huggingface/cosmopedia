# Synthetic data from Web samples

We built several types of synthetic content from seed web samples: textbooks (in narrative or academic tone), blogposts and WikiHow articles.

To select the web samples, we initially clustered 100k samples from a web dataset like [ReFineWeb](https://huggingface.co/datasets/tiiuae/falcon-refinedweb). This resulted into 145 cluters. You can inspect the clusters in this [demo](https://huggingface.co/spaces/HuggingFaceTB/inspect_clusters_free_topics). Then we inferred the clusters of 15M other web samples and used them for the prompts with their topic.

The clustering code can be found in [text-clustering](https://github.com/huggingface/text-clustering?tab=readme-ov-file#cosmopedia-experiments-clustering-of-web-samples-and-topic-labeling) repository. We then excluded 38 clutsers, deemed uneducational, using the scores generated in the clustering, but also after doing some manual inspection of each cluster. We noticed that medium scores weren't always of the topic quality. 

We also tried to infer which generation style would best suit each topic: e.g Mathematics are suitable for textbooks, Beauty & Lifetyle might be suitable for blogposts and DIY for WikiHow articles. However we didn't respect this classification as the prompted LLM seemed to address each topic from interesting different angles when using it with different styles.

Script for classification and filtering (this depends on the clusters you find in your dataset)
```bash
python filter_and_classify_clusters.py
```

Script for building the web prompts:
```bash
python build_web_prompts.py
```
