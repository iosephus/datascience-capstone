
source("config.vars.R")
source("common.R")

require(quanteda)
require(snow)

message("Reading sentences")
sentences = read.corpus.file(sentences.file)

#message("Bringing cluster up")
#cl <- makeCluster(num.cores)
#clusterCall(cl, function() library(quanteda))

tokenize.single.sentence <- function(s) {
    t <- tokenize(s, removePunct = TRUE, removeSeparators = TRUE, removeTwitter = TRUE, removeHyphens = TRUE, simplify=TRUE)
    return(do.call(paste, as.list(toLower(t))))
}

tokenize.sentences <- function(sents) {
    result <- Reduce(c, sapply(sents, tokenize.single.sentence, simplify=TRUE))
    return(result)
}

#clusterExport(cl, "tokenize.single.sentence")
#clusterExport(cl, "tokenize.sentences")

#tokenized.sentences <- Reduce(c, parSapply(cl, FUN = tokenize.sentences, clusterSplit(cl, sentences)))
message("Tokenizing sentences to file")
tokenized.sentences <- tokenize.sentences(sentences)

message("Saving tokenized sentences to file")
conn = file(tokenized.sentences.file, encoding = text.encoding)
writeLines(tokenized.sentences, conn)
close(conn)

#message("Bringing cluster down")
#stopCluster(cl)
