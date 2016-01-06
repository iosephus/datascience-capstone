

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

has_root <- function(ngram, root) {
    tokens <- strsplit(ngram, ' ')[[1]]
    return(all(tokens[1:length(tokens) - 1] == root))
}

create.selector(freq.df, root) {
    sapply(freq.df$ngram, function(x) has_root(x, root))
}

create.model.recur <- function(freq.data, root, current_env) {
    
    assign("freq", freq, current_env)
    assign("children", new.env(), current_env)
}

create.model <- function (freq.data) {
    model = new.env()
    return(model)
}