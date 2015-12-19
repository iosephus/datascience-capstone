
library(tm)
library(SnowballC)
library(wordcloud)

fix.unicode <- function(text.data) {
  result <- text.data
  result <- gsub('â???¦', '.', result)
  result <- gsub('â???"', '-', result)
  result <- gsub('â???"', '-', result)
  result <- gsub('â???T', ''', result)
  result <- gsub('â???o', '"', result)
  result <- gsub('â???[:cntrl:]', '"', result)
  return(result)
}

data.dir <- "C:\\Users\\JoseM\\Projects\\Capstone\\Data\\Coursera-SwiftKey\\final\\en_US"
data.file.blogs <- "en_US.blogs.txt"
data.file.news <- "en_US.news.txt"
data.file.twitter <- "en_US.twitter.txt"

line.limit = 5000
text.encoding = "unknown"
text.language = "en"

content.blogs <- do.call(paste, as.list(readLines(file.path(data.dir, data.file.blogs), n = line.limit, encoding=text.encoding)))
content.news <- do.call(paste, as.list(readLines(file.path(data.dir, data.file.news), n = line.limit, encoding=text.encoding)))
content.twitter <- do.call(paste, as.list(readLines(file.path(data.dir, data.file.twitter), n = line.limit, encoding=text.encoding)))

content.blogs <- fix.unicode(content.blogs)
content.news <- fix.unicode(content.news)
content.twitter <- fix.unicode(content.twitter)

corpus <- Corpus(VectorSource(c(content.blogs, content.news, content.twitter)), readerControl=list(language=text.language))
#corpus <- tm_map(corpus, function(x) iconv(x, to='UTF-8', sub='byte'))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removeWords, stopwords(text.language))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, PlainTextDocument)

# corpus.stemmed <- tm_map(corpus, stemDocument)

dtm <- DocumentTermMatrix(corpus, control=list(wordLengths=c(3,Inf)))
# dtm.st <- DocumentTermMatrix(corpus.stemmed)

# Create cloud of words
dtm2 <- removeSparseTerms(dtm, 0.4)
freq <- sort(colSums(as.matrix(dtm2)), decreasing=TRUE)
words <- names(freq)
wordcloud(words[1:100], freq[1:100])
