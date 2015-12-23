
library(parallel)
library(tidyr)
library(tm)
library(SnowballC)
library(wordcloud)
options( java.parameters = "-Xmx12g" )
library(RWeka)

num.cores = detectCores(all.tests = FALSE, logical = FALSE)

fix.unicode <- function(text.data) {
    result <- text.data
    #result <- gsub('â€¦', '…', result)
    #result <- gsub('â€“', '–', result)
    #result <- gsub('â€”', '–', result)
    #result <- gsub('â€™', '’', result)
    #result <- gsub('â€œ', '“', result)
    #result <- gsub('â€', '”', result)
    result <- iconv(result, "latin1", "ASCII", sub="")
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
  f <- function(x) NGramTokenizer(x, Weka_control(min = n, max = n))
  return(f)
}


get.file.list <- function(dir.path, pattern="*.txt") {
  files <- sapply(list.files(dir.path, pattern="*.txt"), FUN=function(fname) file.path(dir.path, fname))
  return(files)
}

load.corpus.file <- function(path, max.lines=-1, encoding="UTF-8", line.factor=1.0, seed=NULL) {
  con <- file(path, "rb")
  lines <- readLines(con, n = max.lines, encoding=encoding, skipNul=TRUE)
  if (line.factor < 1.0) {
     if (!is.null(seed)) {
         set.seed(seed)
     }
    lines <- sample(lines, round(line.factor * length(lines)), replace=FALSE)
  }
  close(con)
  result <- data.frame(line=seq(1, length(lines)))
  result$content <- lines
  result$file <- basename(path)
  return(result)
}

load.corpus.content <- function(dir.path, max.lines=-1, encoding='unknown', line.factor=1.0, seed=NULL) {
  files <- get.file.list(dir.path, pattern="*.txt")
  contents <- mclapply(files, FUN=function(path) load.corpus.file(path, max.lines, encoding, line.factor, seed), mc.cores=num.cores)
  result <- do.call(rbind, contents)
  rownames(result) <- NULL
  result$file = as.factor(result$file)
  return(result)
}

get.file.size <- function(file.path) {
  info <- file.info(file.path)
  return(info$size)
}

get.file.sizes <- function(file.list) {
  sizes <- lapply(file.list, FUN=get.file.size)
  names(sizes) <- names(file.list)
  result <- gather(data.frame(sizes), key="file", value="size")
  return(result)
}

get.pattern.stats <- function(data, categories=c("[[:alpha:]]", "[[:space:]]", "[[:digit:]]", "[[:punct:]]"), categories.names=c("alphanumeric", "space", "digit", "punctuation")) {
  contents <- data$content
  get.category.counts <- function(pattern) sapply(contents, function (c) sum(gregexpr(pattern, c)[[1]] > 0), USE.NAMES=FALSE)
  categories.counts <- mclapply(categories, FUN=get.category.counts, mc.cores=num.cores)
  names(categories.counts) <- categories.names
  counts <- data.frame(categories.counts)
  counts$nchar <- nchar(contents)
  #counts$file <- data$file
  line.counts <- data.frame(table(data$file))
  names(line.counts) <- c("file", "lines")
  aggregated.counts = aggregate(counts, by=list(file=data$file), FUN=sum)
  result <- merge(aggregated.counts, line.counts)
  selector.cols.numeric <- !grepl("file", names(result))
  total.stats <- data.frame(lapply(result[, selector.cols.numeric], sum))
  total.stats$file = "ALL"
  result <- rbind(result, total.stats)
  result[, categories.names] = result[, categories.names] / result$nchar
  result$other <- 1.0 - apply(result[, categories.names], 1, sum)
  return(result)
}

create.corpus <- function(contents, language="en", remove.stopwords=FALSE) {
  corpus <- Corpus(VectorSource(contents), readerControl=list(language=language))
  corpus <- tm_map(corpus, content_transformer(tolower))
  if (remove.stopwords) {
    corpus <- tm_map(corpus, removeWords, stopwords(language))
  }
  corpus <- tm_map(corpus, remove.punctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, PlainTextDocument)
  return(corpus)
}

count.stopwords <- function(ngram, words=stopwords("en")) {
    result <- sum(strsplit(ngram, " ")[[1]] %in% words)
    return(result)
}

get.ngram.freq <- function(dtm) {
    freq <- colSums(as.matrix(dtm))
    freq.sum = sum(freq)
    result <- data.frame(freq=freq / freq.sum)
    result$ngram <- as.character(names(freq))
    result$stopwords <- as.vector(mcmapply(result$ngram, FUN=count.stopwords, mc.cores=num.cores))
    rownames(result) <- NULL
    result <- result[order(result$freq, decreasing=TRUE), ]
    return(result)
}

ggplotColours <- function(n=6, h=c(0, 360) +15){
    if ((diff(h)%%360) < 1) h[2] <- h[2] - 360/n
    hcl(h = (seq(h[1], h[2], length = n)), c = 100, l = 65)
}

