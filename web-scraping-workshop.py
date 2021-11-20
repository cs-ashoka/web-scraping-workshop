import time
import requests
from bs4 import BeautifulSoup

import pandas as pd
import json

import tabulate 
import pprint

def exportData(title, author, rating, desc, image, reviews):
    if isinstance(title, list):
        assert len(title) == len(author) == len(rating) == len(desc) == len(image), "they all should match in size"

        # create a dataframe
        data = {
            "title": title,
            "author": author,
            "rating": rating,
            "desc": desc,
            "image": image
        }

        df = pd.DataFrame(data)

        # export to csv
        df.to_csv("book_data.csv", index=False)

        for title_ in title:
            # fetch image from url and export to file with name title
            with open(title_ + ".jpg", "wb") as f:
                f.write(requests.get(image).content)

    else:
        # export to json
        data = {
            "title": title,
            "author": author,
            "rating": rating,
            "desc": desc,
            "image": image
        }

        with open(f"{title.replace(':', '-').replace(' ', '-')}.json", "w+") as f:
            json.dump(data, f)

        # export image to title
        with open(title.replace(':', '-').replace(' ', '-') + ".jpg", "wb") as f:
            f.write(requests.get(image).content)
            
        
        reviewsDF = pd.DataFrame(reviews)
        
        reviewsDF.to_csv(f"{title.replace(':', '-').replace('/', '-').replace(' ', '-')}.csv", index=False)


def performNLP(reviews):
    # credits Neil Chowdhury for this function
    import nltk
    from nltk.sentiment import vader
    
    def regressionGraph(reviews):
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        sns.regplot(x = [ x['rating'] for x in reviews ], y = [ x['polarity'] for x in reviews ], line_kws={'color':'red'}).set_title('Polarity vs Rating')
        plt.show()
         
        x = np.array([ x['rating'] for x in reviews ]).reshape((-1, 1))
        y = np.array([ x['polarity'] for x in reviews ]).reshape((-1, 1))
        
        model = LinearRegression().fit(x, y).score(x, y)
        
        print("Regression Coefficient:", model)
        
    
    def polarityScore(reviews):
        sentimentIntensityAnalyzer = vader.SentimentIntensityAnalyzer()
        for review in reviews:
            review["polarity"] = (sentimentIntensityAnalyzer.polarity_scores(review["review"])['compound']+1)*(2)+1
        return reviews

    reviews = polarityScore(reviews)
    
    ratingList = [ review["rating"] for review in reviews ]
    polarityList = [ review["polarity"] for review in reviews ]
    
    averageRating = sum(ratingList)/len(ratingList)
    averagePolarity = sum(polarityList)/len(polarityList)
    
    print(tabulate.tabulate([[averageRating, averagePolarity]], headers=["Average Rating", "Average Polarity"]))
    
    regressionGraph(reviews)


def bsoupScrape(url):
    # important, remember the .text at the end to avoid a parsing error
    pageText = requests.get(url).text
    
    # this requires correct version of bs4 to be installed if using apt-get, if using pip it's all good. html.parser is included
    soup = BeautifulSoup(pageText, 'html.parser')

    # id == id, class == class_, attrs == attrs={"name": "val"}
    bookTitle = soup.find('h1', id='bookTitle').get_text().replace('  ', '').replace('\n', '')

    print(bookTitle)

    bookAuthors = soup.find_all('div', class_='authorName__container')

    bookAuthors_info = []

    for bookAuthor in bookAuthors:
        authorName = bookAuthor.find('span').get_text()
        # find_parent finds the argument tag which is the parent. 
        # you can use .get to get some attribute
        authorLink = bookAuthor.find('span').find_parent("a").get("href")
        bookAuthors_info.append({"name": authorName, "link": authorLink})
    
    pprint.pprint(bookAuthors_info)

    # show nested tag example using reviews
    bookRating = float(soup.find('div', id='bookMeta').find('span', attrs={"itemprop": "ratingValue"}).get_text())

    print(bookRating)

    # show how this one has two, a shorter revealed one, and a longer hidden one. Also show how the id of the span 
    # might be different across different books, so we have to handle it recursively. Do a pprint or a tabulate to show the two elements
    bookDescription = soup.find('div', id='descriptionContainer').find('div', id='description').find_all('span')

    # pprint.pprint(bookDescription)

    # get the last span tag, get the text from it, and remove the whitespaces and text formatting tags. 
    bookDescription = bookDescription[-1].get_text().replace('<i>', ' ').replace('</i>', ' ').replace('\n', ' ').replace('  ', ' ')

    print(bookDescription)

    bookImage = soup.find('img', id='coverImage').get('src')

    print(bookImage)

    
    # finds all the review blocks
    bookReviewBlocks = soup.find_all('div', class_='friendReviews')
    
    bookReviews = list()
        
    for bookReviewBlock in bookReviewBlocks:
        # get the title of the review
        reviewAuthor = bookReviewBlock.find('a', class_='imgcol').get('title').replace('  ', '').replace('\n', '')
    
        # review text was obtained by finding the two spans, one which reveals on reveal.     
        reviewText = bookReviewBlock.find('div', class_='bodycol').find('span', class_='readable').find_all('span')[-1].get_text().replace('<i>', ' ').replace('</i>', ' ').replace('\n', ' ').replace('  ', ' ')
        
        # rating is obtained by counting the number of p10 objects, which contain the full stars
        reviewRating = len(bookReviewBlock.find('div', class_='bodycol').find('div', class_='reviewHeader').find_all('span', class_='p10'))
        
        # append the reviews to the list
        bookReviews.append({"author": reviewAuthor, "review": reviewText, "rating": reviewRating})


    # ooooh what even is NLP?
    performNLP(bookReviews)

    exportData(bookTitle, bookAuthors_info, bookRating, bookDescription, bookImage, bookReviews)


if __name__ == '__main__':
    bsoupScrape('https://www.goodreads.com/book/show/1386986.The_Lost_Ravioli_Recipes_of_Hoboken')
