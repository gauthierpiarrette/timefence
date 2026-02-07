# Timefence Website

This directory contains the landing page for Timefence.

## Structure
- `index.html`: Single-page responsive landing page (Tailwind CSS via CDN).
- `favicon.svg`: Brand icon.

## Deployment (GitHub Pages)

1. Go to your repository Settings > Pages.
2. Source: `Deploy from a branch`.
3. Branch: `main`, Folder: `/website` (if allowed) or root.
   
   *Note: If GitHub Pages requires the root directory, you might need to push the contents of this folder to a separate `gh-pages` branch.*

   **Recommended workflow for monorepo:**
   Use a GitHub Action to deploy this folder to `gh-pages` branch.

## Development
Just open `index.html` in your browser. No build step required.
