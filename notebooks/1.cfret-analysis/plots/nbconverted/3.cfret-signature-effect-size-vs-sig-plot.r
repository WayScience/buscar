suppressPackageStartupMessages({library(arrow)
library(dplyr)
library(ggplot2)
library(tidyr)
library(viridis)
library(RColorBrewer)
library(patchwork)
library(ggside)
library(IRdisplay)})

# setting signature stats paths
signatures_stats_path <- file.path("../results/signatures/signature_importance.csv")
shuffle_signatures_stats_path <- file.path("../results/signatures/shuffle_signature_importance.csv")

if (!file.exists(signatures_stats_path)) {
  stop(paste("File not found:", signatures_stats_path))
}
if (!file.exists(shuffle_signatures_stats_path)) {
  stop(paste("File not found:", shuffle_signatures_stats_path))
}

# setting output path for the generated plot
sig_plot_output_dir = file.path("./figures")
if (!dir.exists(sig_plot_output_dir)) {
  dir.create(sig_plot_output_dir, showWarnings = FALSE, recursive = TRUE)
}

# Load both signature stats files and label each by shuffled status
sig_stats_df <- read.csv(signatures_stats_path)
sig_stats_df$data_type <- "Non-shuffled"

shuffle_stats_df <- read.csv(shuffle_signatures_stats_path)
shuffle_stats_df$data_type <- "Shuffled"

# Combine into a single dataframe and set factor order so non-shuffled appears on top
combined_df <- rbind(sig_stats_df, shuffle_stats_df)
combined_df$data_type <- factor(combined_df$data_type, levels = c("Non-shuffled", "Shuffled"))

# Keep channel independent from compartment; use source column when available
if (!("channel" %in% colnames(combined_df))) {
  combined_df$channel <- "no-channel"
}
combined_df$channel <- ifelse(is.na(combined_df$channel) | combined_df$channel == "", "no-channel", combined_df$channel)

head(combined_df)

# Configure plot dimensions — enlarged for publication
height <- 10
width <- 32
options(repr.plot.width = width, repr.plot.height = height)

# Generate color palettes for the different stratifications
n_compartments <- length(unique(combined_df$compartment))
compartment_palette <- brewer.pal(max(3, min(n_compartments, 8)), "Dark2")

# Handle channel palette: use a distinct palette for channels, and handle NA/missing
unique_channels <- sort(unique(combined_df$channel))
n_channels <- length(unique_channels)
channel_palette_base <- colorRampPalette(brewer.pal(8, "Set1"))(n_channels)
names(channel_palette_base) <- unique_channels

# Set a shared Y-axis limit across both datasets so plots are directly comparable
y_max <- max(combined_df$neg_log10_p_value[is.finite(combined_df$neg_log10_p_value)], na.rm = TRUE) * 1.1

make_stratified_plots <- function(df, show_yside = TRUE, title_suffix = "") {

  # remove infinite values for plotting
  df <- df[is.finite(df$neg_log10_p_value), ]

  # Replace NA/empty channels with "no-channel"
  df$channel[is.na(df$channel) | df$channel == ""] <- "no-channel"

  # Update palette for this specific dataframe view if "no-channel" exists
  local_channel_palette <- channel_palette_base
  if ("no-channel" %in% df$channel) {
    local_channel_palette["no-channel"] <- "gray60"
  }

  # Add a newline before the suffix to prevent cutoff
  display_suffix <- if(title_suffix != "") paste0("\n", title_suffix) else ""

  # Base theme for all panels
  base_theme <- theme_minimal(base_size = 39.2) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 39.2, lineheight = 0.8),
      axis.title = element_text(size = 33.6, face = "bold"),
      axis.text = element_text(size = 28),
      axis.text.x = element_text(size = 28, angle = 90, vjust = 0.5, hjust = 1),
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 30.8),
      legend.text = element_text(size = 25.2),
      legend.box.spacing = grid::unit(0.24, "cm"),
      legend.margin = margin(3.3, 3.3, 3.3, 3.3),
      panel.grid.minor = element_blank(),
      ggside.panel.scale = 0.2,
      ggside.panel.spacing = grid::unit(0.12, "cm")
    )

  # Panel 1: Compartments
  p1 <- ggplot(df, aes(x = ks_stat, y = neg_log10_p_value, color = compartment)) +
    geom_point(size = 2, alpha = 0.6) +
    geom_xsidedensity(aes(y = after_stat(ndensity), fill = compartment), alpha = 0.3, color = NA, position = "identity") +
    scale_color_manual(values = compartment_palette) +
    scale_fill_manual(values = compartment_palette, guide = "none") +
    scale_y_continuous(limits = c(0, y_max), oob = scales::squish) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "gray40") +
    labs(x = "KS statistic", y = "-log10(FDR-corrected-p-value)", title = "Compartment", color = "Compartment") +
    guides(color = guide_legend(override.aes = list(size = 6, alpha = 1))) +
    base_theme

  # Panel 2: Channels
  p2 <- ggplot(df, aes(x = ks_stat, y = neg_log10_p_value, color = channel)) +
    geom_point(size = 2, alpha = 0.6) +
    geom_xsidedensity(aes(y = after_stat(ndensity), fill = channel), alpha = 0.3, color = NA, position = "identity") +
    scale_color_manual(values = local_channel_palette) +
    scale_fill_manual(values = local_channel_palette, guide = "none") +
    scale_y_continuous(limits = c(0, y_max), oob = scales::squish) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "gray40") +
    labs(x = "KS statistic", y = NULL, title = "Channel", color = "Channel") +
    guides(color = guide_legend(override.aes = list(size = 6, alpha = 1))) +
    base_theme

  # Panel 3: Signatures
  p3 <- ggplot(df, aes(x = ks_stat, y = neg_log10_p_value, color = signature)) +
    geom_point(size = 2, alpha = 0.6) +
    geom_xsidedensity(aes(y = after_stat(ndensity), fill = signature), alpha = 0.3, color = NA, position = "identity") +
    scale_color_manual(
      values = c("off" = "gray60", "on" = "#E41A1C"),
      labels = c("off" = "off-morphology", "on" = "on-morphology")
    ) +
    scale_fill_manual(values = c("off" = "gray60", "on" = "#E41A1C"), guide = "none") +
    scale_y_continuous(limits = c(0, y_max), oob = scales::squish) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "gray40") +
    labs(x = "KS statistic", y = NULL, title = "Signature", color = "Signature") +
    guides(color = guide_legend(override.aes = list(size = 6, alpha = 1))) +
    base_theme

  if (show_yside) {
    p1 <- p1 + geom_ysidedensity(aes(x = after_stat(ndensity), fill = compartment), alpha = 0.3, color = NA, position = "identity")
    p2 <- p2 + geom_ysidedensity(aes(x = after_stat(ndensity), fill = channel), alpha = 0.3, color = NA, position = "identity")
    p3 <- p3 + geom_ysidedensity(aes(x = after_stat(ndensity), fill = signature), alpha = 0.3, color = NA, position = "identity")
  }

  combined_plot <- (p1 | p2 | p3) +
    plot_annotation(
      title = paste0("Feature significance and effect size ", display_suffix),
      theme = theme(
        plot.title = element_text(size = 44.8, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 33.6, hjust = 0.5, margin = margin(b = 20))
      )
    )

  return(combined_plot)
}

# Non-shuffled plot (with y-side distributions)
non_shuffled_plot <- make_stratified_plots(combined_df[combined_df$data_type == "Non-shuffled", ], show_yside = TRUE)
output_png_path <- file.path(sig_plot_output_dir, "cfret_stratified_significance_plots.png")
ggsave(output_png_path, non_shuffled_plot, width = width, height = height, dpi = 300, bg = "white")

# Shuffled plot (as requested)
shuffled_plot <- make_stratified_plots(combined_df[combined_df$data_type == "Shuffled", ], show_yside = TRUE, title_suffix = "(shuffled)")
shuffled_output_png_path <- file.path(sig_plot_output_dir, "shuffled_cfret_stratified_significance_plots.png")
ggsave(shuffled_output_png_path, shuffled_plot, width = width, height = height, dpi = 300, bg = "white")

cat("Saved:", output_png_path, "\n")
cat("Saved:", shuffled_output_png_path, "\n")

non_shuffled_plot
shuffled_plot
