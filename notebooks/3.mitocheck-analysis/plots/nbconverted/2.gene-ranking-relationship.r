suppressPackageStartupMessages({
    library(arrow)
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(pheatmap)
    library(grid)
})

options(warn = -1)

# Input data directory
results_dir <- normalizePath('../results/moa_analysis', mustWork = TRUE)

# Output directory for this notebook's figures/tables
gene_rel_dir <- normalizePath(file.path(getwd(), 'all-plots', 'gene-ranking-relationships'), mustWork = FALSE)
dir.create(gene_rel_dir, recursive = TRUE, showWarnings = FALSE)

# Load real and shuffled BuSCaR outputs
moa_results_df <- read_parquet(file.path(results_dir, 'original_mitocheck_moa_analysis_results.parquet')) %>% as_tibble()
shuffled_moa_results_df <- read_parquet(file.path(results_dir, 'shuffled_mitocheck_moa_analysis_results.parquet')) %>% as_tibble()

# Recompute rank within each phenotype to avoid duplicate/missing rank values.
# Lower on/off score is better, so rows are sorted accordingly before row_number().
rerank_by_profile <- function(input_df) {
  input_df %>%
    arrange(ref_profile, is.na(on_score), on_score, is.na(off_score), off_score, treatment) %>%
    group_by(ref_profile) %>%
    mutate(rank = row_number()) %>%
    ungroup()
}

moa_results_df <- rerank_by_profile(moa_results_df)
shuffled_moa_results_df <- rerank_by_profile(shuffled_moa_results_df)

head(moa_results_df)

# Keep rows where both on_score and off_score are present
prepare_df <- function(results_df) {
  results_df %>%
    as.data.frame() %>%
    filter(!is.na(on_score), !is.na(off_score))
}

df <- prepare_df(moa_results_df)
shuf_df <- prepare_df(shuffled_moa_results_df)

build_rank_bundle <- function(input_df) {
  rank_pivot <- input_df %>%
    select(ref_profile, treatment, rank) %>%
    pivot_wider(names_from = ref_profile, values_from = rank)

  rank_pivot_complete <- rank_pivot %>% drop_na()

  rank_pivot_reranked <- rank_pivot_complete %>%
    select(-treatment) %>%
    mutate(across(everything(), ~ rank(.x, ties.method = 'average')))

  # Use sparse, profile-wise rank matrix for pairwise correlations.
  # This avoids collapsing to ~1 treatment for shuffled data after complete-case filtering.
  rank_matrix_sparse <- rank_pivot %>%
    select(-treatment) %>%
    mutate(across(everything(), ~ rank(.x, ties.method = 'average', na.last = 'keep')))

  list(
    rank_pivot = rank_pivot,
    rank_pivot_complete = rank_pivot_complete,
    rank_pivot_reranked = rank_pivot_reranked,
    rank_matrix_sparse = rank_matrix_sparse,
    n_subjects = nrow(rank_pivot_reranked),
    k_raters = ncol(rank_pivot_reranked),
    profile_names = colnames(rank_matrix_sparse)
  )
}

orig_bundle <- build_rank_bundle(df)
shuf_bundle <- build_rank_bundle(shuf_df)

# options for rendering the figure larger
height <- 12
width <- 14
options(repr.plot.width = width, repr.plot.height = height)

compute_consistency <- function(bundle) {
  R_complete <- as.matrix(bundle$rank_pivot_reranked)

  if (nrow(R_complete) > 1 && ncol(R_complete) > 1) {
    rank_sums <- rowSums(R_complete)
    S <- sum((rank_sums - mean(rank_sums))^2)
    W <- (12 * S) / (bundle$k_raters^2 * (bundle$n_subjects^3 - bundle$n_subjects))
    chi2_stat <- bundle$k_raters * (bundle$n_subjects - 1) * W
    chi2_pval <- pchisq(chi2_stat, df = bundle$n_subjects - 1, lower.tail = FALSE)
    w_label <- sprintf('%.3f', W)
  } else {
    W <- NA_real_
    chi2_stat <- NA_real_
    chi2_pval <- NA_real_
    w_label <- 'NA (insufficient complete-case treatments)'
  }

  # Pairwise correlation on sparse rank matrix preserves shuffled information.
  # This avoids collapsing to ~1 treatment for shuffled data after complete-case filtering.
  R_sparse <- as.matrix(bundle$rank_matrix_sparse)
  corr_matrix <- cor(R_sparse, method = 'spearman', use = 'pairwise.complete.obs')

  diag(corr_matrix) <- 1
  corr_long <- as.data.frame(as.table(corr_matrix))
  colnames(corr_long) <- c('y', 'x', 'rho')
  corr_long$x <- factor(corr_long$x, levels = bundle$profile_names)
  corr_long$y <- factor(corr_long$y, levels = rev(bundle$profile_names))

  list(
    W = W,
    w_label = w_label,
    chi2_stat = chi2_stat,
    chi2_pval = chi2_pval,
    corr_matrix = corr_matrix,
    corr_long = corr_long,
    n_subjects = bundle$n_subjects,
    k_raters = bundle$k_raters
  )
}

plot_corr_heatmap <- function(consistency, title_suffix = '') {
  ggplot(consistency$corr_long, aes(x = x, y = y, fill = rho)) +
    geom_tile(color = 'white', linewidth = 0.3) +
    geom_text(aes(label = ifelse(is.na(rho), 'NA', sprintf('%.2f', rho))), size = 6.3, color = '#222222') +
    scale_fill_gradient2(
      low = '#d62728',
      mid = '#f7f7f7',
      high = '#1f77b4',
      midpoint = 0,
      limits = c(-1, 1),
      na.value = 'grey80'
    ) +
    labs(
      title = sprintf(
        "Cross-phenotype Spearman rank correlation ρ %s\nKendall's W = %s  (complete-case n = %d, %d phenotypic states)",
        title_suffix, consistency$w_label, consistency$n_subjects, consistency$k_raters
      ),
      x = NULL,
      y = NULL,
      fill = 'Spearman rank correlation'
    ) +
    theme_minimal(base_size = 20) +
    theme(
      plot.title = element_text(face = 'bold', size = 25, hjust = 0.5, color = '#1a1a2e'),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 20, color = '#222222'),
      axis.text.y = element_text(size = 20, color = '#222222'),
      legend.title = element_text(face = 'bold', size = 22, color = '#1a1a2e'),
      legend.text = element_text(size = 20, color = '#222222'),
      panel.grid = element_blank()
    )
}

save_and_print_corr <- function(consistency, file_name, title_suffix = '', label = 'original') {
  cat(sprintf("[%s] Kendall's W = %s\n", label, consistency$w_label))
  cat(sprintf('[%s] complete-case n = %d, k = %d\n', label, consistency$n_subjects, consistency$k_raters))

  if (!is.na(consistency$chi2_stat)) {
    cat(sprintf('[%s] χ²(%d) = %.4f,  p = %.4e\n', label, consistency$n_subjects - 1, consistency$chi2_stat, consistency$chi2_pval))
  }

  p_corr <- plot_corr_heatmap(consistency, title_suffix)
  out_path <- file.path(gene_rel_dir, file_name)
  ggsave(out_path, p_corr, width = width, height = height, dpi = 300, bg = 'white')
  cat(sprintf('[%s] Saved -> %s\n', label, out_path))
  print(p_corr)
}

orig_consistency <- compute_consistency(orig_bundle)
shuf_consistency <- compute_consistency(shuf_bundle)

save_and_print_corr(
  orig_consistency,
  'profile_rank_consistency.png',
  '',
  'original'
)

save_and_print_corr(
  shuf_consistency,
  'shuffled_profile_rank_consistency.png',
  ' (shuffled)',
  'shuffled'
)

# options for rendering the figure larger
height <- 12
width <- 14
options(repr.plot.width = width, repr.plot.height = height)

save_corr_clustermap <- function(consistency, file_name, title_suffix = '', label = 'original') {
  corr_clean <- consistency$corr_matrix
  corr_clean[is.na(corr_clean)] <- 0

  anno_numbers <- ifelse(is.na(consistency$corr_matrix), 'NA', sprintf('%.2f', consistency$corr_matrix))
  dim(anno_numbers) <- dim(consistency$corr_matrix)

  pheat <- pheatmap(
    corr_clean,
    color = colorRampPalette(c('#d62728', '#f7f7f7', '#1f77b4'))(256),
    breaks = seq(-1, 1, length.out = 257),
    cluster_rows = TRUE,
    cluster_cols = TRUE,
    display_numbers = anno_numbers,
    number_color = '#222222',
    fontsize = 20,
    fontsize_row = 20,
    fontsize_col = 20,
    fontsize_number = 17,
    border_color = '#e0e0e0',
    main = sprintf(
      "Cross-phenotype Spearman ρ — hierarchical clustering%s\nKendall's W = %s  (complete-case n = %d, k = %d)",
      title_suffix, consistency$w_label, consistency$n_subjects, consistency$k_raters
    )
  )

  out_path <- file.path(gene_rel_dir, file_name)
  png(out_path, width = 14, height = 14, units = 'in', res = 300)
  grid::grid.newpage()
  grid::grid.draw(pheat$gtable)
  dev.off()
  cat(sprintf('[%s] Saved -> %s\n', label, out_path))
}

save_corr_clustermap(
  orig_consistency,
  'profile_rank_consistency_clustermap.png',
  '',
  'original'
)

save_corr_clustermap(
  shuf_consistency,
  'shuffled_profile_rank_consistency_clustermap.png',
  ' (shuffled)',
  'shuffled'
)

# Compute Kendall's W (and chi-square p-value) from a complete-case rank matrix
compute_kendall_w_from_matrix <- function(rank_matrix_complete) {
  n_subjects <- nrow(rank_matrix_complete)
  k_raters <- ncol(rank_matrix_complete)

  if (n_subjects <= 1 || k_raters <= 1) {
    return(list(
      W = NA_real_,
      p_value = NA_real_,
      chi2_stat = NA_real_,
      n_subjects = n_subjects,
      k_raters = k_raters
    ))
  }

  row_rank_sums <- rowSums(rank_matrix_complete)
  S <- sum((row_rank_sums - mean(row_rank_sums))^2)
  W <- (12 * S) / (k_raters^2 * (n_subjects^3 - n_subjects))

  chi2_stat <- k_raters * (n_subjects - 1) * W
  p_value <- pchisq(chi2_stat, df = n_subjects - 1, lower.tail = FALSE)

  list(
    W = W,
    p_value = p_value,
    chi2_stat = chi2_stat,
    n_subjects = n_subjects,
    k_raters = k_raters
  )
}

# Build the complete-case rank matrix from the original data and compute observed W
R_real <- as.matrix(orig_bundle$rank_pivot_reranked)

real_stats <- compute_kendall_w_from_matrix(R_real)


# Permutation settings
set.seed(0)
n_permutations <- 5000

# Null model: for each phenotype column, independently shuffle which gene receives
# which rank. This keeps the within-phenotype rank distribution intact but destroys
# the cross-phenotype concordance signal -- the standard null for Kendall's W.
perm_W <- replicate(n_permutations, {
  R_perm <- apply(R_real, 2, sample, replace = FALSE)
  if (is.vector(R_perm)) R_perm <- matrix(R_perm, ncol = ncol(R_real))
  compute_kendall_w_from_matrix(R_perm)$W
})

perm_W <- perm_W[!is.na(perm_W)]
empirical_p_real <- if (length(perm_W) > 0) mean(perm_W >= real_stats$W) else NA_real_
z_vs_perm <- if (length(perm_W) > 1 && sd(perm_W) > 0) (real_stats$W - mean(perm_W)) / sd(perm_W) else NA_real_

summary_tbl <- tibble::tibble(
  comparison = c('real', 'perm_null_mean', 'perm_null_sd', 'empirical_p_real_vs_perm', 'zscore_real_vs_perm'),
  value = c(
    real_stats$W,
    ifelse(length(perm_W) > 0, mean(perm_W), NA_real_),
    ifelse(length(perm_W) > 1, sd(perm_W), NA_real_),
    empirical_p_real,
    z_vs_perm
  )
)

print(summary_tbl)

summary_out <- file.path(gene_rel_dir, 'kendall_w_real_permutation_summary.csv')
readr::write_csv(summary_tbl, summary_out)
cat(sprintf('Saved -> %s\n', summary_out))


# Plot the permutation-null distribution with the real W as a reference line
plot_df <- tibble::tibble(W = perm_W)

p_perm <- ggplot(plot_df, aes(x = W)) +
  geom_histogram(bins = 40, fill = '#6baed6', color = 'white', alpha = 0.95) +
  geom_vline(xintercept = real_stats$W, color = '#d62728', linewidth = 1.2) +
  annotate('text', x = real_stats$W, y = Inf, label = sprintf('real W = %.3f', real_stats$W), vjust = 1.5, hjust = -0.05, color = '#d62728', size = 6.3) +
  scale_x_continuous(limits = c(NA, 0.5)) +
  labs(
    title = "Kendall's W: Real vs Permutation Null (label permutation)",
    subtitle = sprintf('Permutation null (n = %d) | empirical p(real >= null) = %s', n_permutations, ifelse(is.na(empirical_p_real), 'NA', sprintf('%.4f', empirical_p_real))),
    x = "Kendall's W under label-permutation null",
    y = 'Frequency'
  ) +
  theme_minimal(base_size = 20) +
  theme(
    plot.title = element_text(face = 'bold', size = 25),
    plot.subtitle = element_text(size = 18)
  )

perm_plot_out <- file.path(gene_rel_dir, 'kendall_w_permutation_null_histogram.png')
ggsave(perm_plot_out, p_perm, width = 11, height = 7, dpi = 300, bg = 'white')
cat(sprintf('Saved -> %s\n', perm_plot_out))

print(p_perm)

# Summarize phenotype specificity as mean off-diagonal Spearman correlation per profile
# (real data only -- uses the correlation matrices computed in Analysis 2).
profile_specificity <- tibble::tibble(profile = colnames(orig_consistency$corr_matrix)) %>%
  mutate(
    mean_rho_real = sapply(profile, function(p) {
      v <- orig_consistency$corr_matrix[p, setdiff(colnames(orig_consistency$corr_matrix), p)]
      mean(v, na.rm = TRUE)
    })
  ) %>%
  arrange(mean_rho_real)

spec_out <- file.path(gene_rel_dir, 'profile_specificity_real.csv')
readr::write_csv(profile_specificity, spec_out)
cat(sprintf('Saved -> %s\n', spec_out))

profile_specificity
