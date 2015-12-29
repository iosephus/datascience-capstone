

read.corpus.file <- function(path) {
    con <- file(path, "rb")
    result <- readLines(con, encoding=text.encoding, skipNul=TRUE)
    close(con)
    return(result)
}

split.vector <- function(v, n) {
    result <- split(v, ceiling(seq_along(v) / n))
    return(result)
}