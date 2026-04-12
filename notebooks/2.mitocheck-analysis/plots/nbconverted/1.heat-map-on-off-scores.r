suppressPackageStartupMessages({
    library(arrow)
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(pheatmap)
    library(viridisLite)
    library(stringr)
    library(stats)
    library(grid)
    library(gridExtra)
})

options(warn = -1)

truncate_palette <- function(palette_fun, min_val = 0.15, max_val = 1.0, n = 256) {
  vals <- seq(min_val, max_val, length.out = n)
  palette_fun(n)[pmax(1, pmin(n, round(vals * n)))]
}

sig_stars <- function(p) {
  if (is.na(p)) return('n.s.')
  if (p < 0.001) return('***')
  if (p < 0.01) return('**')
  if (p < 0.05) return('*')
  'n.s.'
}

# setting result dir
results_dir <- normalizePath('../results/logo_analysis', mustWork = TRUE)

# setting output
output_dir <- normalizePath(file.path(getwd(), 'all-plots', 'heatmap'), mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# loadding in moa results
moa_results_df <- read_parquet(file.path(results_dir, 'original_mitocheck_logo_analysis_results.parquet')) %>% as_tibble()
shuffled_moa_results_df <- read_parquet(file.path(results_dir, 'shuffled_mitocheck_logo_analysis_results.parquet')) %>% as_tibble()

# rerank treatment to remove duplicate ranks (nulls ranked last)
rerank <- function(input_df) {
  input_df %>%
    arrange(target, is.na(on_buscar_scores), on_buscar_scores, perturbation) %>%
    group_by(target) %>%
    mutate(rank = row_number()) %>%
    ungroup()
}

# re rranking based on only on-buscar scores
moa_results_df <- rerank(moa_results_df)
shuffled_moa_results_df <- rerank(shuffled_moa_results_df)

prepare_df <- function(results_df) {
  results_df %>%
    as.data.frame() %>%
    filter(!is.na(on_buscar_scores), !is.na(off_buscar_scores))
}

df <- prepare_df(moa_results_df)
shuf_df <- prepare_df(shuffled_moa_results_df)

profiles <- sort(unique(df$target))
n_profiles <- length(profiles)

# Install ggnewscale if not already available
if (!requireNamespace("ggnewscale", quietly = TRUE)) {
  install.packages("ggnewscale", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(ggnewscale)
  library(scales)
})

# ── Constants ─────────────────────────────────────────────────────────────────
N_SHOW   <- Inf   # show all genes
NCOLS    <- 8
TILE_W   <- 0.30
TILE_H   <- 0.85
ON_X     <- TILE_W * 1.5
OFF_X    <- TILE_W * 2.8
BAR_FILL <- "#5b9bd5"

# ── Data preparation ──────────────────────────────────────────────────────────
make_plot_df <- function(raw_df) {
  raw_df %>%
    filter(!is.na(on_buscar_scores), !is.na(off_buscar_scores)) %>%
    group_by(target) %>%
    arrange(rank) %>%
    slice_head(n = N_SHOW) %>%
    ungroup() %>%
    arrange(target, desc(rank)) %>%
    mutate(
      prop_scaled = -proportion * TILE_W,
      gene_key = factor(
        paste0(target, "___", perturbation),
        levels = unique(paste0(target, "___", perturbation))
      )
    )
}

# ── Plot builder ──────────────────────────────────────────────────────────────
build_ranking_plot <- function(plot_df, title_str) {
  colorbar_guide <- function(ttl) {
    guide_colorbar(
      title          = ttl,
      title.position = "top",
      title.hjust    = 0.5,
      barwidth       = unit(4.5, "cm"),
      barheight      = unit(0.4, "cm")
    )
  }

  ggplot(plot_df) +
    geom_col(
      aes(x = prop_scaled, y = gene_key),
      fill  = BAR_FILL,
      width = TILE_H,
      alpha = 0.88
    ) +
    geom_vline(xintercept = 0, colour = "grey70", linewidth = 0.25) +
    geom_tile(
      aes(x = ON_X, y = gene_key, fill = on_buscar_scores),
      width  = TILE_W,
      height = TILE_H
    ) +
    scale_fill_gradient(
      low      = "#e8f1fb",
      high     = "#2c7fb8",
      na.value = "#e8f1fb",
      guide    = colorbar_guide("On score")
    ) +
    new_scale_fill() +
    geom_tile(
      aes(x = OFF_X, y = gene_key, fill = off_buscar_scores),
      width  = TILE_W,
      height = TILE_H
    ) +
    scale_fill_gradient(
      low      = "#fff0df",
      high     = "#d95f0e",
      na.value = "#fff0df",
      guide    = colorbar_guide("Off score")
    ) +
    scale_x_continuous(
      breaks = c(-TILE_W, -TILE_W / 2, 0, ON_X, OFF_X),
      labels = c("100 %", "50 %", "0 %", "On", "Off"),
      expand = expansion(add = c(0.02, 0.08))
    ) +
    scale_y_discrete(labels = function(x) sub(".*___", "", x)) +
    facet_wrap(~ target, scales = "free_y", ncol = NCOLS) +
    labs(
      title = title_str,
      x     = "\u2190 Proportion  |  Score \u2192",
      y     = NULL
    ) +
    theme_minimal(base_size = 15) +
    theme(
      strip.text       = element_text(face = "bold", lineheight = 10, size = 25, colour = "#111111",
                                       margin = margin(5, 5,5, 5)),
      strip.background = element_blank(),
      strip.clip       = "off",
      axis.text.y      = element_text(face = "italic", size = 22, colour = "#111111",
                                       margin = margin(r = 2)),
      axis.text.x      = element_text(size = 18,  colour = "#555555",
                                       angle = 90, vjust = 0.5, hjust = 1,
                                       margin = margin(t = 2)),
      axis.title.x     = element_text(size = 18, colour = "#444444",
                                       margin = margin(t = 5)),
      axis.ticks       = element_blank(),
      panel.grid       = element_blank(),
      panel.border     = element_rect(colour = "#cccccc", fill = NA, linewidth = 0.25),
      panel.spacing.x  = unit(0.8, "lines"),
      panel.spacing.y  = unit(0.6, "lines"),
      legend.position  = "bottom",
      legend.direction = "horizontal",
      legend.title     = element_text(size = 20, face = "bold"),
      legend.text      = element_text(size = 18),
      legend.spacing.x = unit(1.0, "cm"),
      legend.box       = "horizontal",
      legend.margin    = margin(t = 6),
      plot.title       = element_text(face = "bold", size = 40, hjust = 0.5,
                                       margin = margin(b = 10)),
      plot.background  = element_rect(fill = "white", colour = NA),
      plot.margin      = margin(10, 10, 10, 10)
    )
}

# ── Render helper ─────────────────────────────────────────────────────────────
render_ranking_plot <- function(raw_df, out_path, title_str) {
  plot_df    <- make_plot_df(raw_df)
  n_profiles <- length(unique(plot_df$target))
  n_rows     <- ceiling(n_profiles / NCOLS)
  max_genes  <- max(table(plot_df$target))

  row_h  <- max_genes * 0.30 + 2.8
  fig_h  <- n_rows * row_h + 2.0
  fig_w  <- NCOLS * 5.2

  p <- build_ranking_plot(plot_df, title_str)

  ggsave(
    filename  = out_path, plot = p,
    width     = fig_w, height = fig_h,
    units     = "in", dpi = 300, bg = "white",
    limitsize = FALSE
  )
  cat(sprintf("Saved \u2192 %s\n", out_path))

  if (requireNamespace("IRdisplay", quietly = TRUE)) {
    IRdisplay::display_png(file = out_path)
  }
}

# ── Original ──────────────────────────────────────────────────────────────────
render_ranking_plot(
  df,
  file.path(output_dir, "gene_rankings_phenotypic_state.png"),
  "Gene rankings per phenotypic state"
)

# ── Shuffled ──────────────────────────────────────────────────────────────────
render_ranking_plot(
  shuf_df,
  file.path(output_dir, "shuffled_gene_rankings_phenotypic_state.png"),
  "Gene rankings per phenotypic state (shuffled)"
)

# ── Subset heatmap: selected phenotypes ───────────────────────────────────────
SELECTED_PHENOTYPES <- c(
  "Apoptosis", "Anaphase", "Metaphase",
  "Prometaphase", "MetaphaseAlignment", "SmallIrregular"
)

# Filter original data to the chosen phenotypes only
df_subset <- df %>%
  filter(target %in% SELECTED_PHENOTYPES)

# Output path for the subset plot (same heatmap dir)
subset_output_dir <- normalizePath(file.path(getwd(), 'all-plots', 'heatmap'), mustWork = FALSE)
dir.create(subset_output_dir, recursive = TRUE, showWarnings = FALSE)

# Temporarily override NCOLS to suit 6 panels (3 columns × 2 rows)
NCOLS_orig <- NCOLS
NCOLS <- 6

render_ranking_plot(
  df_subset,
  file.path(subset_output_dir, "gene_rankings_selected_phenotypes.png"),
  "Gene rankings per phenotype"
)

# Restore global NCOLS
NCOLS <- NCOLS_orig

# ── Complement heatmap: all phenotypes excluding the selected subset ───────────
EXCLUDED_PHENOTYPES <- c(
  "Apoptosis", "Anaphase", "Metaphase",
  "Prometaphase", "MetaphaseAlignment", "SmallIrregular"
)

# Keep every phenotype not in the excluded set
df_complement <- df %>%
  filter(!target %in% EXCLUDED_PHENOTYPES)

# Output path for the complement plot (same heatmap dir)
complement_output_dir <- normalizePath(file.path(getwd(), 'all-plots', 'heatmap'), mustWork = FALSE)
dir.create(complement_output_dir, recursive = TRUE, showWarnings = FALSE)

# Temporarily override NCOLS to suit 10 panels (5 columns × 2 rows)
NCOLS_orig <- NCOLS
NCOLS <- 5

render_ranking_plot(
  df_complement,
  file.path(complement_output_dir, "gene_rankings_remaining_phenotypes.png"),
  "Gene rankings"
)

# Restore global NCOLS
NCOLS <- NCOLS_orig
