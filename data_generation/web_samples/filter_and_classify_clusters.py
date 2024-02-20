import argparse
import pandas as pd
from collections import Counter
from datasets import load_dataset, Dataset


# re-classifying each topic into the most adequate class: textbook, blogpost, or wikihow
classifications = {
    "textbook": [
        'International Relations and Politics',
        'Product Marketing and Design',
        'Digital Imaging and Photography',
        'Computer Science',
        'Economics and Finance',
        'Pharmaceutical manufacturing and technology',
        'Real Estate & Investment',
        'Business and Entrepreneurship',
        'Astronomy and Astrophysics',
        'Finance and Investment',
        'Computer Antivirus Software and Security',
        'Healthcare and Operations Management',
        'Technology and Computer Science',
        'Computer Programming and Web Development',
        'Taxation and Finance',
        'Human Resources / Organizational Management',
        'Computer Hardware and Graphics Cards',
        'Marketing and Business Strategies',
        'Digital Marketing and Business',
        'Audio Equipment and Home Theater Systems',
        'HIV Treatment and Care',
        'Legal Studies and Public Policy',
        'Legal Studies / Law',
        'Jewelry Design and Manufacturing',
        'Biochemistry and Molecular Biology',
        'Insurance',
        'Energy and Environmental Policy',
        'Data Privacy and Protection',
        'International Relations and Conflict',
        'Entomology and Apiculture',
        'Loans and Mortgages',
        'Public Transit and Transportation',
        'International Relations and Current Events',
        'Politics and Government',
        'Political Science',
        'Genetics and Mental Health',
        'Public Administration and Policy',
        'Technology and Consumer Electronics',
        'Computer Security & Privacy',
        'Online Platforms & Web Technologies',
        'Human Resources and Education',
        'Sports and Education',
        'Lighting Design and Technology',
        'Medicine',
        'Cryptocurrency and Blockchain Technology',
        'Mental Health Counseling',
        'Geography and Weather',
        'Leadership and Education',
        'Infant Feeding and Child Development',
        'Molecular Biology and Genetics',
        'Energy and Natural Resources',
        'Mental Health and Therapy',
        'Business and Management',
        'Legal Services and Issues',
        'Christian Theology and Spirituality',
        'Personal Finance and Investments',
        'Psychology',
        'Healthcare & Medical Services',
        'Watchmaking and Horology',
        'Online Chat Platforms and Data Privacy',
        'Waste Management and Recycling'
    ],
    "blogpost": [
        'Health and Lifestyle',
        'Physical Fitness and Health',
        'Music',
        'Fiction and Fantasy Writing',
        'Literature and Creative Writing',
        'Arts and Crafts',
        'Education',
        'Education and Youth Development',
        'Writing and Storytelling',
        'Hair Care and Styling',
        'Automotive Parts and Accessories',
        'Astrology',
        'Culinary Arts and Beverages',
        'Events and Community Happenings',
        'Cooking and Baking',
        'Online Dating & Relationships',
        'Career Development and Job Opportunities',
        'Cosmetic Surgery and Body Modifications',
        'Skincare and Beauty Products',
        'Addiction and Mental Illness',
        'Visual Arts and Art Appreciation',
        'Pets and Pet Care',
        'Personal Development and Empowerment',
        'Video Games',
        'Hair Care',
        'Nutrition and Health',
        'Fashion & Apparel',
        'Travel',
        'Performing Arts',
        'Cannabis and CBD Products',
        'Wine & Winemaking',
        'Cooking and Recipes'
    ],
    "wikihow": [
        'Dentistry',
        'Football/Soccer',
        'Cleaning and Maintenance',
        'American Football',
        'Baseball',
        'Recreational Fishing',
        'Public Safety and Emergency Response',
        'Ice Hockey',
        'Professional Basketball/NBA',
        'Home Improvement and Maintenance',
        'Tennis',
        'Professional Wrestling and Sports Entertainment',
        'Cricket',
        'Gun Control and Violence',
        'Fire Incidents',
        'Electric Vehicles and Battery Technology',
        'Christianity and Theology'
    ]
}
remove = ['Online Gambling and Casinos',
    'Reality Television and Celebrity Gossip',
    'Explicit Adult Content',
    'Fashion & Accessories',
    'Solicitation and Prostitution',
    'Adult Entertainment and Webcam Sites',
    'Clothing & Fashion',
    'Firearms and Accessories',
    'Marketing and Sales Promotions', 
    'Cookies and Privacy Policies', 
    'Fragrances and Personal Care Products',
    'Obituaries and Personal Profiles', 
    'Motor Sports',
    'Death',
    'Male Enhancement Products and Supplements',
    'Political Science',
    'Heating',
    'Anabolic Steroids and',
    'Combat Sports',
    'Events and Community Happenings',
    'Footwear and Fashion',
    'Furniture Design and Sales',
    'Entertainment & Media',
    'Real Estate & Property Management',
    'Weight Loss and Body Contouring',
    'Moving Services and Logistics'
    'Transportation and City Planning',
    'Home Decoration and Furniture',
    'Events and Conferences',
    'Cosmetics and Beauty Products',
    'Mattresses and Sleep',
    'Soap & Skincare Products',
    'E-commerce and Online Shopping',
    'Optical Equipment and Accessories',
    ]

new_dict = {v[i]: k for k, v in classifications.items() for i in range(len(v))}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters_dataset", type=str, default="HuggingFaceTB/web_clusters")
    parser.add_argument("--user", type=str, default="HuggingFaceTB")
    return parser.parse_args()


def extract_category(example):
    summary = example["summary"]
    category = summary.split(". Educational")[0].strip()
    score = summary.split(" Educational score: ")[1].strip()
    return {"category": category, "educational_score": score}


def add_generation_type(x):
    topic = x["category"]
    try:
        generation_type = new_dict[topic]
    except:
        print(f"{topic} not in keep list")
        generation_type = "blogpost"
    return {"generation_type": generation_type}

args = get_args()
print("Loading web samples (after the clustering)...")
ds_1 = load_dataset(args.clusters_dataset, split="train")
print(ds_1)

print("Converting to dataframe...")
full_df = ds_1.to_pandas().explode("examples")

full_df.sort_values(by=['cluster_id'], inplace=True)

print("Full df info...")
print(full_df.head())
print(full_df.info())

print("Convert to HF dataset...")
final_ds = Dataset.from_pandas(full_df)
final_ds = final_ds.map(extract_category)
print("HF dataset:")
print(final_ds)

print("Filter out bad topics...")
ds_keep = final_ds.filter(lambda x: x["category"] not in remove, num_proc=64)
print(f"Size after dropping low quality clusters: {len(ds_keep)}={len(ds_keep)*100/len(final_ds):.2f}% of the original dataset")

print("Add generation type...")
ds_keep = ds_keep.map(add_generation_type, num_proc=24)
print(Counter(ds_keep["generation_type"]))

print("Retrieve textbooks...")
textbooks = ds_keep.filter(lambda x: x["generation_type"] == "textbook")
print(textbooks)
print("Retrieve wikihow...")
wikihow = ds_keep.filter(lambda x: x["generation_type"] == "wikihow")
print(wikihow)
print("Retrieve blopgpot...")
blogpost = ds_keep.filter(lambda x: x["generation_type"] == "blogpost")
print(blogpost)

print("Pushing to hub ...")
textbooks.push_to_hub(f"{args.user}/fw2_as_textbook", private=True)
wikihow.push_to_hub(f"{args.user}/fw2_as_wikihow", private=True)
blogpost.push_to_hub(f"{args.user}/fw2_as_blogpost", private=True)
print("Done!")