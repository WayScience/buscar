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
combined_df$channel <- sapply(strsplit(combined_df$feature, "_"), `[`, 1)

head(combined_df)

# Configure plot dimensions — enlarged for publication
height <- 12
width <- 25
options(repr.plot.width = width, repr.plot.height = height)

# Generate a color palette for the different channels
n_channels <- length(unique(combined_df$channel))
dark2_palette <- brewer.pal(max(3, min(n_channels, 8)), "Dark2")

# Set a shared Y-axis limit across both datasets so plots are directly comparable
y_max <- max(combined_df$neg_log10_p_value[is.finite(combined_df$neg_log10_p_value)], na.rm = TRUE) * 1.1

make_plots <- function(df, show_yside = TRUE, title_suffix = "") {
  # Add a newline before the suffix to prevent cutoff
  display_suffix <- if(title_suffix != "") paste0("\n", title_suffix) else ""

  plot_channel <- ggplot(df, aes(x = ks_stat, y = neg_log10_p_value, color = channel)) +
    geom_point(size = 3, alpha = 0.7) +
    geom_xsidedensity(aes(y = after_stat(ndensity), fill = channel), alpha = 0.4, color = NA, position = "identity") +
    scale_color_manual(values = dark2_palette) +
    scale_fill_manual(values = dark2_palette, guide = "none") +
    scale_y_continuous(
      limits = c(0, y_max),
      oob = scales::squish,
      expand = expansion(mult = c(0.02, 0.05))
    ) +
    scale_xsidey_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "gray40", linewidth = 0.6) +
    labs(
      x = "KS statistic (effect size)",
      y = "-log10(FDR-corrected p-value)",
      title = paste0("Feature significance by compartment", display_suffix),
      color = "Compartment"
    ) +
    theme_minimal(base_size = 31) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 36, lineheight = 0.8), # Slightly reduced size and added lineheight
      axis.title = element_text(size = 35, face = "bold"),
      axis.text = element_text(size = 28),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 28),
      strip.text = element_text(face = "bold", size = 26),
      strip.background = element_rect(fill = "gray92", color = NA),
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 32),
      legend.text = element_text(size = 28),
      legend.key.size = unit(1.2, "lines"),
      legend.background = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(35, 15, 25, 25), # Increased top margin for multi-line titles
      ggside.panel.scale = 0.2,
      ggside.axis.text.x = element_text(size = 17, angle = 90, vjust = 0.5, hjust = 1),
      ggside.axis.text.y = element_text(size = 17)
    )

  plot_significant <- ggplot(df, aes(x = ks_stat, y = neg_log10_p_value, color = signature)) +
    geom_point(size = 3, alpha = 0.7) +
    geom_xsidedensity(aes(y = after_stat(ndensity), fill = signature), alpha = 0.4, color = NA, position = "identity") +
    scale_color_manual(
      values = c("off" = "gray60", "on" = "#E41A1C"),
      labels = c("off" = "off-morphological", "on" = "on-morphological")
    ) +
    scale_fill_manual(
      values = c("off" = "gray60", "on" = "#E41A1C"),
      guide = "none"
    ) +
    scale_y_continuous(
      limits = c(0, y_max),
      oob = scales::squish,
      expand = expansion(mult = c(0.02, 0.05))
    ) +
    scale_xsidey_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "gray40", linewidth = 0.6) +
    labs(
      x = "KS statistic (effect size)",
      y = "-log10(FDR-corrected p-value)",
      title = paste0("Feature significance by signature", display_suffix),
      color = "Signature"
    ) +
    theme_minimal(base_size = 31) +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 36, lineheight = 0.8),
      axis.title = element_text(size = 35, face = "bold"),
      axis.text = element_text(size = 28),
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 28),
      strip.text = element_text(face = "bold", size = 26),
      strip.background = element_rect(fill = "gray92", color = NA),
      legend.position = "right",
      legend.title = element_text(face = "bold", size = 32),
      legend.text = element_text(size = 28),
      legend.key.size = unit(1.2, "lines"),
      legend.background = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(35, 25, 25, 15),
      ggside.panel.scale = 0.2,
      ggside.axis.text.x = element_text(size = 17, angle = 90, vjust = 0.5, hjust = 1),
      ggside.axis.text.y = element_text(size = 17)
    )

  if (show_yside) {
    plot_channel <- plot_channel +
      geom_ysidedensity(aes(x = after_stat(ndensity), fill = channel), alpha = 0.4, color = NA, position = "identity") +
      scale_ysidex_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"))
    plot_significant <- plot_significant +
      geom_ysidedensity(aes(x = after_stat(ndensity), fill = signature), alpha = 0.4, color = NA, position = "identity") +
      scale_ysidex_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"))
  }

  plot_channel + plot_significant
}

# Non-shuffled plot (with y-side distributions)
non_shuffled_plot <- make_plots(combined_df[combined_df$data_type == "Non-shuffled", ], show_yside = TRUE)
output_png_path <- file.path(sig_plot_output_dir, "cfret_signature_significance_plots.png")
ggsave(output_png_path, non_shuffled_plot, width = width, height = height, dpi = 300, bg = "white")

# Shuffled plot (without y-side distributions and with " (shuffled)" title suffix)
shuffled_plot <- make_plots(combined_df[combined_df$data_type == "Shuffled", ], show_yside = FALSE, title_suffix = "(shuffled)")
shuffled_output_png_path <- file.path(sig_plot_output_dir, "shuffled_cfret_signature_significance_plots.png")
ggsave(shuffled_output_png_path, shuffled_plot, width = width, height = height, dpi = 300, bg = "white")

cat("Saved:", output_png_path, "\n")
cat("Saved:", shuffled_output_png_path, "\n")

non_shuffled_plot
shuffled_plot
