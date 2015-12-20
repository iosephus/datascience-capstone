
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
  result <- gsub('â???[[:cntrl:]]', '"', result)
  return(result)
}

remove.punctuation <- function(x) {
  result <- x
  result <- gsub('[[:punct:]]', "", result)
  #result <- gsub('[-!"#$%&()*+,./:;<=>?@[\\\\]\\^_`\\{|\\}~]', "", result)
  #result <- gsub("([^a-zA-Z0-9])'([^a-zA-Z])", "\\1\\2", result)
  return(result)
}

NGramTokenizerFuncBuilder <- function(n) {
  f <- function(x) NGramTokenizer(x, Weka_control(min = n, max = n, delimiters = delimiters))
  return(f)
}

data.dir <- "C:\\Users\\JoseM\\Projects\\Capstone\\Data\\Coursera-SwiftKey\\final\\en_US"
data.file.blogs <- "en_US.blogs.txt"
data.file.news <- "en_US.news.txt"
data.file.twitter <- "en_US.twitter.txt"

line.limit = -1
text.encoding = "unknown"
text.language = "en"

content.blogs <- readLines(file.path(data.dir, data.file.blogs), n = line.limit, encoding=text.encoding)
content.news <- readLines(file.path(data.dir, data.file.news), n = line.limit, encoding=text.encoding)
content.twitter <- readLines(file.path(data.dir, data.file.twitter), n = line.limit, encoding=text.encoding)

content.blogs <- fix.unicode(content.blogs)
content.news <- fix.unicode(content.news)
content.twitter <- fix.unicode(content.twitter)

corpus <- Corpus(VectorSource(c(content.blogs, content.news, content.twitter)), readerControl=list(language=text.language))
#corpus <- tm_map(corpus, function(x) iconv(x, to='UTF-8', sub='byte'))
corpus <- tm_map(corpus, tolower)
#corpus <- tm_map(corpus, removeWords, stopwords(text.language))
corpus <- tm_map(corpus, remove.punctuation)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, PlainTextDocument)

# corpus.stemmed <- tm_map(corpus, stemDocument)

dtm <- DocumentTermMatrix(corpus, control=list(wordLengths=c(3,25)))
# dtm.st <- DocumentTermMatrix(corpus.stemmed)


# Create cloud of words
dtm2 <- removeSparseTerms(dtm, 0.4)

freq <- sort(colSums(as.matrix(dtm)), decreasing=TRUE)
words <- names(freq)
wordcloud(words[1:100], freq[1:100])

dtm.bigrams <- DocumentTermMatrix(corpus, control = list(tokenize = NGramTokenizerFuncBuilder(2)))
freq.bigrams <- sort(colSums(as.matrix(dtm.bigrams)), decreasing=TRUE)
bigrams <- names(freq.bigrams)
wordcloud(bigrams[1:50], freq.bigrams[1:50])


dtm.trigrams <- DocumentTermMatrix(corpus, control = list(tokenize = NGramTokenizerFuncBuilder(3)))
freq.trigrams <- sort(colSums(as.matrix(dtm.trigrams)), decreasing=TRUE)
trigrams <- names(freq.trigrams)
wordcloud(trigrams[1:30], freq.trigrams[1:30])
