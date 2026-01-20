suppressPackageStartupMessages({
    library(tidyverse)
    library(arrow)
    library(jsonlite)
    library(networkD3)
    library(htmlwidgets)
    library(RColorBrewer)
    library(viridis)
    library(webshot)
    library(plotly)
    library(IRdisplay)
})

set.seed(42)

results_dir <- file.path(".", "results")
plots_dir <- file.path(results_dir, "plots", "sankey")
cluster_dir <- file.path(results_dir, "clusters")
pca_dir <- file.path(results_dir, "pca")

dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(cluster_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(pca_dir, showWarnings = FALSE, recursive = TRUE)

cluster_labels_path <- file.path(cluster_dir, "all_sc_cfret_optimized_cluster_labels.parquet")
sc_profile_path <- file.path("..", "..", "0.download-data", "data", "sc-profiles",
                              "cfret", "localhost230405150001_sc_feature_selected.parquet")

if (!file.exists(cluster_labels_path)) {
    stop("Cluster labels file not found.")
}
if (!file.exists(sc_profile_path)) {
    stop("Single-cell profile file not found.")
}

cat("Setup complete\n")

cluster_labels_df <- read_parquet(cluster_labels_path, as_data_frame = TRUE)
cluster_labels_df <- cluster_labels_df %>%
    mutate(Metadata_cluster_id = str_extract(Metadata_cluster_id, "[^_]+$"))

sc_profiles_df <- read_parquet(sc_profile_path, as_data_frame = TRUE)
metadata_cols <- colnames(sc_profiles_df)[str_detect(colnames(sc_profiles_df), "^Metadata_")]
sc_metadata_df <- sc_profiles_df %>%
    select(all_of(metadata_cols))

sankey_df <- cluster_labels_df %>%
    inner_join(sc_metadata_df, by = "Metadata_cell_id")

cat(sprintf("Loaded %d cells\n", nrow(sankey_df)))
cat(sprintf("Clusters: %d\n", n_distinct(sankey_df$Metadata_cluster_id)))
cat(sprintf("Cell types: %s\n", paste(unique(sankey_df$Metadata_cell_type), collapse = ", ")))

head(sankey_df, 5)

cell_type_colors <- c(
    "Failing" = "#E41A1C",
    "Healthy" = "#377EB8",
    "Rescue" = "#4DAF4A"
)

treatment_colors <- brewer.pal(8, "Dark2")
cluster_palette <- "Set3"
link_opacity <- 0.3

base_font_size <- 14
label_font_size <- 12
title_font_size <- 16
legend_font_size <- 11
font_family <- "Arial"

cat("Color palettes configured\n")

sankey_df <- sankey_df %>%
    mutate(cell_type_treatment = paste(Metadata_cell_type, Metadata_treatment, sep = "_"))

celltype_cluster_counts <- sankey_df %>%
    group_by(cell_type_treatment, Metadata_cluster_id) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(cell_type_treatment, Metadata_cluster_id)

sources <- unique(celltype_cluster_counts$cell_type_treatment)
targets <- unique(celltype_cluster_counts$Metadata_cluster_id)

nodes <- data.frame(
    name = c(sources, targets),
    group = c(rep("source", length(sources)), rep("target", length(targets))),
    stringsAsFactors = FALSE
)

links <- celltype_cluster_counts %>%
    mutate(
        source = match(cell_type_treatment, nodes$name) - 1,
        target = match(Metadata_cluster_id, nodes$name) - 1,
        value = count
    ) %>%
    select(source, target, value)

node_colors <- c(
    rep("#E8998D", length(sources)),
    viridis(length(targets), option = "D", alpha = 0.8)
)

sankey1 <- sankeyNetwork(
    Links = links,
    Nodes = nodes,
    Source = "source",
    Target = "target",
    Value = "value",
    NodeID = "name",
    fontSize = label_font_size,
    fontFamily = font_family,
    nodeWidth = 30,
    nodePadding = 10,
    iterations = 32,
    sinksRight = TRUE,
    height = 600,
    width = 1000
)

sankey1

html_path1 <- file.path(plots_dir, "celltype_treatment_cluster_sankey.html")
saveWidget(sankey1, html_path1, selfcontained = TRUE)

cat(sprintf("\nSaved to: %s\n", html_path1))
cat(sprintf("Sources: %d, Targets: %d, Flows: %d, Total cells: %d\n",
            length(sources), length(targets), nrow(links), sum(links$value)))

sankey_df <- sankey_df %>%
    mutate(phenotype_well_treatment = paste(Metadata_cell_type, Metadata_Well,
                                             Metadata_treatment, sep = "_"))

well_cluster_counts <- sankey_df %>%
    group_by(phenotype_well_treatment, Metadata_cluster_id) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(phenotype_well_treatment, Metadata_cluster_id)

sources_well <- unique(well_cluster_counts$phenotype_well_treatment)
targets_well <- unique(well_cluster_counts$Metadata_cluster_id)

nodes_well <- data.frame(
    name = c(sources_well, targets_well),
    group = c(rep("source", length(sources_well)), rep("target", length(targets_well))),
    stringsAsFactors = FALSE
)

links_well <- well_cluster_counts %>%
    mutate(
        source = match(phenotype_well_treatment, nodes_well$name) - 1,
        target = match(Metadata_cluster_id, nodes_well$name) - 1,
        value = count
    ) %>%
    select(source, target, value)

sankey2 <- sankeyNetwork(
    Links = links_well,
    Nodes = nodes_well,
    Source = "source",
    Target = "target",
    Value = "value",
    NodeID = "name",
    fontSize = label_font_size - 2,
    fontFamily = font_family,
    nodeWidth = 25,
    nodePadding = 8,
    iterations = 32,
    sinksRight = TRUE,
    height = 800,
    width = 1200
)

sankey2

html_path2 <- file.path(plots_dir, "phenotype_well_treatment_cluster_sankey.html")
saveWidget(sankey2, html_path2, selfcontained = TRUE)

cat(sprintf("\nSaved to: %s\n", html_path2))
cat(sprintf("Sources: %d, Targets: %d, Flows: %d\n",
            length(sources_well), length(targets_well), nrow(links_well)))

treatment_cluster_counts <- sankey_df %>%
    group_by(Metadata_treatment, Metadata_cluster_id) %>%
    summarise(count = n(), .groups = "drop") %>%
    arrange(Metadata_treatment, Metadata_cluster_id)

sources_treat <- unique(treatment_cluster_counts$Metadata_treatment)
targets_treat <- unique(treatment_cluster_counts$Metadata_cluster_id)

nodes_treat <- data.frame(
    name = c(sources_treat, targets_treat),
    group = c(rep("source", length(sources_treat)), rep("target", length(targets_treat))),
    stringsAsFactors = FALSE
)

links_treat <- treatment_cluster_counts %>%
    mutate(
        source = match(Metadata_treatment, nodes_treat$name) - 1,
        target = match(Metadata_cluster_id, nodes_treat$name) - 1,
        value = count
    ) %>%
    select(source, target, value)

sankey3 <- sankeyNetwork(
    Links = links_treat,
    Nodes = nodes_treat,
    Source = "source",
    Target = "target",
    Value = "value",
    NodeID = "name",
    fontSize = label_font_size,
    fontFamily = font_family,
    nodeWidth = 35,
    nodePadding = 15,
    iterations = 32,
    sinksRight = TRUE,
    height = 600,
    width = 1000
)

sankey3

html_path3 <- file.path(plots_dir, "treatment_cluster_sankey.html")
saveWidget(sankey3, html_path3, selfcontained = TRUE)

cat(sprintf("\nSaved to: %s\n", html_path3))
cat(sprintf("Sources: %d, Targets: %d, Flows: %d\n",
            length(sources_treat), length(targets_treat), nrow(links_treat)))

well_cluster_contribution <- sankey_df %>%
    group_by(Metadata_Well, Metadata_cluster_id) %>%
    summarise(n_cells = n(), .groups = "drop") %>%
    arrange(Metadata_Well, Metadata_cluster_id)

well_cluster_pivot <- well_cluster_contribution %>%
    pivot_wider(
        names_from = Metadata_cluster_id,
        values_from = n_cells,
        values_fill = 0,
        names_prefix = "cluster_"
    )

well_metadata <- sankey_df %>%
    group_by(Metadata_Well) %>%
    summarise(
        cell_type = first(Metadata_cell_type),
        treatment = first(Metadata_treatment),
        total_cells = n(),
        .groups = "drop"
    )

well_cluster_pivot <- well_cluster_pivot %>%
    left_join(well_metadata, by = "Metadata_Well") %>%
    select(Metadata_Well, cell_type, treatment, total_cells, everything())

cluster_cols <- colnames(well_cluster_pivot)[str_detect(colnames(well_cluster_pivot), "^cluster_")]

well_cluster_pct <- well_cluster_pivot %>%
    mutate(across(all_of(cluster_cols), ~ . / total_cells * 100, .names = "{.col}_pct"))

contrib_table_path <- file.path(cluster_dir, "well_cluster_contribution.csv")
write_csv(well_cluster_pct, contrib_table_path)

cat("\n--- Summary ---\n")
cat(sprintf("Total cells: %d\n", nrow(sankey_df)))
cat(sprintf("Unique wells: %d\n", n_distinct(sankey_df$Metadata_Well)))
cat(sprintf("Clusters: %d\n", n_distinct(sankey_df$Metadata_cluster_id)))

cells_per_well <- sankey_df %>% group_by(Metadata_Well) %>% summarise(n = n())
cat(sprintf("Cells per well: %.1f ± %.1f\n", mean(cells_per_well$n), sd(cells_per_well$n)))

cells_per_cluster <- sankey_df %>% group_by(Metadata_cluster_id) %>% summarise(n = n())
cat(sprintf("Cells per cluster: %.1f ± %.1f\n", mean(cells_per_cluster$n), sd(cells_per_cluster$n)))

head(well_cluster_pct, 10)

suppressPackageStartupMessages({
    library(ComplexHeatmap)
    library(circlize)
    library(grid)
})

pct_cols <- colnames(well_cluster_pct)[str_detect(colnames(well_cluster_pct), "^cluster_.*_pct$")]
cluster_names <- str_remove(pct_cols, "_pct$")

heatmap_matrix <- well_cluster_pct %>%
    select(all_of(pct_cols)) %>%
    as.matrix()

colnames(heatmap_matrix) <- str_remove(cluster_names, "^cluster_")
rownames(heatmap_matrix) <- paste0(
    well_cluster_pct$Metadata_Well,
    " (", well_cluster_pct$cell_type, ", ",
    well_cluster_pct$treatment, ")"
)

col_fun <- colorRamp2(
    c(0, max(heatmap_matrix, na.rm = TRUE) / 2, max(heatmap_matrix, na.rm = TRUE)),
    c("#FFFFCC", "#FD8D3C", "#BD0026")
)

title_fontsize <- 14
label_fontsize <- 10
legend_fontsize <- 10

unique_cell_types <- unique(well_cluster_pct$cell_type)
unique_treatments <- unique(well_cluster_pct$treatment)
n_treatments <- length(unique_treatments)

cell_type_palette <- c(
    "Failing" = "#E41A1C",
    "failing" = "#E41A1C",
    "Healthy" = "#377EB8",
    "healthy" = "#377EB8",
    "Rescue" = "#4DAF4A",
    "rescue" = "#4DAF4A"
)
cell_type_palette <- cell_type_palette[names(cell_type_palette) %in% unique_cell_types]

if (n_treatments <= 8) {
    treatment_palette <- setNames(
        brewer.pal(max(3, n_treatments), "Dark2")[1:n_treatments],
        unique_treatments
    )
} else {
    treatment_palette <- setNames(
        viridis(n_treatments, option = "D"),
        unique_treatments
    )
}

row_ha <- rowAnnotation(
    `Cell Type` = well_cluster_pct$cell_type,
    `Treatment` = well_cluster_pct$treatment,
    col = list(
        `Cell Type` = cell_type_palette,
        `Treatment` = treatment_palette
    ),
    annotation_name_gp = gpar(fontsize = label_fontsize, fontface = "bold"),
    annotation_legend_param = list(
        `Cell Type` = list(title_gp = gpar(fontsize = legend_fontsize, fontface = "bold")),
        `Treatment` = list(title_gp = gpar(fontsize = legend_fontsize, fontface = "bold"))
    )
)

ht <- Heatmap(
    heatmap_matrix,
    name = "% of Well",
    col = col_fun,
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    clustering_distance_rows = "euclidean",
    clustering_distance_columns = "euclidean",
    clustering_method_rows = "complete",
    clustering_method_columns = "complete",
    left_annotation = row_ha,
    row_names_gp = gpar(fontsize = 8),
    column_names_gp = gpar(fontsize = label_fontsize, fontface = "bold"),
    column_title = "Well Distribution Across Clusters",
    column_title_gp = gpar(fontsize = title_fontsize, fontface = "bold"),
    heatmap_legend_param = list(
        title = "% of Well\nin Cluster",
        title_gp = gpar(fontsize = legend_fontsize, fontface = "bold"),
        labels_gp = gpar(fontsize = legend_fontsize - 1),
        legend_height = unit(4, "cm"),
        grid_width = unit(0.5, "cm")
    ),
    cell_fun = function(j, i, x, y, width, height, fill) {
        if (heatmap_matrix[i, j] > 5) {
            grid.text(sprintf("%.1f", heatmap_matrix[i, j]), x, y,
                     gp = gpar(fontsize = 7, col = "white"))
        }
    },
    use_raster = TRUE,
    raster_quality = 5
)

options(repr.plot.width = 12, repr.plot.height = 10)

draw(ht,
     heatmap_legend_side = "right",
     annotation_legend_side = "right",
     merge_legend = TRUE)

png_path <- file.path(plots_dir, "well_cluster_heatmap.png")
png(png_path, width = 12, height = 10, units = "in", res = 300)
draw(ht,
     heatmap_legend_side = "right",
     annotation_legend_side = "right",
     merge_legend = TRUE)
dev.off()

cat(sprintf("\nSaved to: %s\n", png_path))
