## Synthetic generations from OpenStax

To build prompts from the OpenStax, the dataset with the course iutline and introductions is avilable [here](https://huggingface.co/datasets/HuggingFaceTB/openstax_paragraphs). We generate textbooks for 4 different audiences: young children, middle school students, professionals and researchers and college students. Each prompt was carefully tailored based on the target audience.  
````
python ./build_openstax_prompts.py 
```