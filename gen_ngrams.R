
source("config.vars.R")
source("common.R")

require(quanteda)
require(snow)

message("Retrieving file list")
file.list = list.files(corpus.dir, corpus.file.pattern, full.names=TRUE)

message("Reading files")
text.lines = Reduce(c, as.vector(lapply(file.list, FUN=read.corpus.file)))

message("Bringing cluster up")
cl <- makeCluster(num.cores)
clusterCall(cl, function() library(quanteda))

message("Splitting work")
chunks <- clusterSplit(cl, text.lines)

message("Tokenizing sentences")
sentences <- parSapply(cl, chunks, FUN = function(x) tokenize(x, what="sentence"))

for (piece in split.vector(text.lines, 3)) {
    sentences <- f(cl, piece, function(x) tokenize(x, what="sentence"))
}

sentences <- lapply(split.vector(text.lines, 3), FUN = function (x) f(cl, x, function(x) tokenize(x, what="sentence")))

message("Releasing memory")
rm(text.lines)
rm(chunks)

tokenize.single.sentence <- function(s) {
    t <- tokenize(s, removePunct = TRUE, removeSeparators = TRUE, removeTwitter = TRUE, removeHyphens = TRUE, simplify=TRUE)
    return(c(token.sentence.start, t, token.sentence.end))
}

tokenize.sentences <- function(sents) {
    result <- Reduce(c, sapply(sents, tokenize.sentence, simplify=TRUE))
    return(result)
} 

unigrams <- Reduce(c, parSapply(cl, FUN = tokenize.sentences, clusterSplit(cl, sentences)))

message("Bringing cluster down")
stopCluster(cl)
