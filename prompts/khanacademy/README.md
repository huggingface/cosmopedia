# KhanAcademy
## Code adapted from https://github.com/rand-net/khan-dl

## Run script to download list of courses
```bash
# install requirements
pip install -r khan_dl/requirements.txt
# run downloader with all courses
python khan_dl/main.py -a
```

output will be saved on `khan_courses.json`

You can then use `generate_textbooks.py` to build the textbook generation prompts.
[TODO]: add code for updated prompts fo Cosmopedia