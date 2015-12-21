
library(tidyr)
library(tm)
library(SnowballC)
library(wordcloud)

fix.unicode <- function(text.data) {
  result <- text.data
  result <- gsub('â€¦', '…', result)
  result <- gsub('â€“', '–', result)
  result <- gsub('â€”', '–', result)
  result <- gsub('â€™', '’', result)
  result <- gsub('â€œ', '“', result)
  result <- gsub('â€[[:cntrl:]]', '”', result)
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


get.file.list <- function(dir.path, pattern="*.txt") {
  files <- sapply(list.files(data.dir, pattern="*.txt"), FUN=function(fname) file.path(dir.path, fname))
  return(files)
}

load.corpus.file <- function(path, max.lines=-1, encoding="UTF-8") {
  lines <- readLines(path, n = max.lines, encoding=encoding)
  result <- data.frame(line=seq(1, length(lines)), content=lines)
  result$file <- basename(path)
  result$content <- as.character(result$content)
  return(result)
}

load.corpus.content <- function(dir.path, max.lines=-1, encoding='unknown') {
  files <- get.file.list(dir.path, pattern="*.txt")
  contents <- lapply(files, FUN=function(path) load.corpus.file(path, max.lines, encoding))
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

get.row.stats <- function(row, categories=c("[[:alpha:]]", "[[:blank:]]", "[[:digit:]]", "[[:punct:]]")) {
  cat.counts <- lapply(categories, function (pattern) sum(grepl(pattern, row$content)))
  result <- data.frame(cat.counts)
  return(results)
}

get.pattern.stats <- function(data, categories=c("[[:alpha:]]", "[[:space:]]", "[[:digit:]]", "[[:punct:]]"), categories.names=c("alphanumeric", "space", "digit", "punctuation")) {
  contents <- data$content
  get.category.counts <- function(pattern) sapply(contents, function (c) sum(gregexpr(pattern, c)[[1]] > 0), USE.NAMES=FALSE)
  categories.counts <- lapply(categories, FUN=get.category.counts)
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

create.corpus <- function(contents, language="en") {
  corpus <- Corpus(VectorSource(contents), readerControl=list(language=language))
  #corpus <- tm_map(corpus, function(x) iconv(x, to='UTF-8', sub='byte'))
  corpus <- tm_map(corpus, tolower)
  #corpus <- tm_map(corpus, removeWords, stopwords(text.language))
  corpus <- tm_map(corpus, remove.punctuation)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, PlainTextDocument)
  return(corpus)
}
