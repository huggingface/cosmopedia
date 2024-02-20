# Stories from UltraChat and OpenHermes

We build several types of stories: educational stories for young children, stories involving morals and principles, stories involving problem solving, and posts found on forums and reddit. The prompts are based on seed samples from [UltraChat](https://huggingface.co/datasets/stingning/ultrachat) and [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5). 

We only use the "Questions about the world" [subset](https://huggingface.co/datasets/HuggingFaceTB/ultrachat_questions_about_world)) of UltraChat. For OpenHermes we filter out non English instruction in OpenHermes and remove categories and sources that wouldn't be suitable for stories, the filtered dataset is available [here](https://huggingface.co/datasets/HuggingFaceTB/openhermes_filtered).

To run the filtering
```bash
python filter_openhermes.py
```

To build the prompts
```bash
python build_openhermes_stories_prompts.py --run_all_styles
python build_ultrachat_stories_prompts.py --run_all_styles
```