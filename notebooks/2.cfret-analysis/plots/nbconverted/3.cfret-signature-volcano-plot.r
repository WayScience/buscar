suppressPackageStartupMessages({library(arrow)
library(dplyr)
library(ggplot2)
library(tidyr)
library(viridis)
library(RColorBrewer)
library(patchwork)
library(IRdisplay)})


# setting signature stats path
signatures_stats_path <- file.path("../results/signatures/signature_importance.csv")
if (!file.exists(signatures_stats_path)) {
  stop(paste("File not found:", signatures_stats_path))
}

# setting output path for the generated plot
sig_plot_output_dir = file.path("./figures")
if (!dir.exists(sig_plot_output_dir)) {
  dir.create(sig_plot_output_dir, showWarnings = FALSE, recursive = TRUE)
}

# load feature space config signatures_stats
sig_stats_df <- read.csv(signatures_stats_path)
head(sig_stats_df)

# Configure plot dimensions for a side-by-side layout
height <- 8
width <- 16
options(repr.plot.width = width, repr.plot.height = height)

# Extract the base feature categorizations (channels/compartments like Cells, Cytoplasm, Nuclei)
# We assume the channel is the first part of the feature string separated by "_"
sig_stats_df$channel <- sapply(strsplit(sig_stats_df$feature, "_"), `[`, 1)

# Generate a color palette for the different channels
n_channels <- length(unique(sig_stats_df$channel))
# brewer.pal requires at least 3 colors to avoid warnings
dark2_palette <- brewer.pal(max(3, min(n_channels, 8)), "Dark2")

# Set consistent Y-axis limits across both plots
# Multiply by 1.1 to provide headroom above the most significant point
y_max <- max(sig_stats_df$neg_log10_p_value[is.finite(sig_stats_df$neg_log10_p_value)], na.rm = TRUE) * 1.1

# Generate Panel 1: Feature significance colored by imaging channel
plot_channel <- ggplot(sig_stats_df, aes(x = ks_stat, y = neg_log10_p_value, color = channel)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = dark2_palette) +
  scale_y_continuous(limits = c(0, y_max), expand = expansion(mult = c(0.02, 0.05))) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "gray40", linewidth = 0.5) +
  labs(
    x = "KS statistic (effect size)",
    y = "-log10(FDR-corrected p-value)",
    title = "Feature significance by channel",
    color = "Channel"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 13),
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  )

# Generate Panel 2: Feature significance colored by signature classification (on vs off)
plot_significant <- ggplot(sig_stats_df, aes(x = ks_stat, y = neg_log10_p_value, color = signature)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(
    # Using consistent terminology: on-morphological and off-morphological signatures
    values = c("off" = "gray60", "on" = "#E41A1C"),
    labels = c("off" = "off-morphological", "on" = "on-morphological")
  ) +
  scale_y_continuous(limits = c(0, y_max), expand = expansion(mult = c(0.02, 0.05))) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "gray40", linewidth = 0.5) +
  labs(
    x = "KS statistic (effect size)",
    y = "-log10(FDR-corrected p-value)",
    title = "Feature significance by signature",
    color = "Signature"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 18),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 13),
    legend.text = element_text(size = 11),
    panel.grid.minor = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  )

# Combine the two generated plots side-by-side using patchwork
combined_plot <- plot_channel + plot_significant

# Save the combined visualization to the output directory
output_png_path <- file.path(sig_plot_output_dir, "cfret_signature_significance_plots.png")
ggsave(output_png_path, combined_plot, width = 16, height = 8, dpi = 300, bg = "white")

# Display the plot in the notebook
combined_plot
