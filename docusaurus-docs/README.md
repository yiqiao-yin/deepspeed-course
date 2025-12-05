# DeepSpeed Course Documentation

This directory contains the Docusaurus-based documentation site for the DeepSpeed Course.

**Live Site:** https://yiqiao-yin.github.io/deepspeed-course/

## Quick Start

### Prerequisites

- Node.js 18.0 or higher
- npm or yarn

### Local Development

```bash
# Navigate to docs directory
cd docusaurus-docs

# Install dependencies
npm install

# Start development server
npm start
```

The site will be available at http://localhost:3000/deepspeed-course/

### Build

```bash
# Create production build
npm run build

# Test the production build locally
npm run serve
```

## Directory Structure

```
docusaurus-docs/
├── docs/                    # Documentation markdown files
│   ├── intro.md            # Homepage/introduction
│   ├── getting-started/    # Getting started guides
│   ├── tutorials/          # Tutorial content
│   │   ├── basic/         # Basic examples (01-04)
│   │   ├── intermediate/  # Intermediate examples
│   │   ├── huggingface/   # HuggingFace integration
│   │   └── multimodal/    # Video/audio training
│   ├── guides/            # Deployment guides
│   └── reference/         # Configuration reference
├── src/                    # Custom React components
│   ├── css/               # Custom styles
│   └── pages/             # Custom pages (homepage)
├── static/                 # Static assets
│   └── img/               # Images and logo
├── docusaurus.config.js    # Main configuration
├── sidebars.js            # Navigation structure
└── package.json           # Dependencies
```

## Adding New Documentation

### New Tutorial

1. Create a markdown file in the appropriate directory:
   ```bash
   # For a new basic tutorial
   touch docs/tutorials/basic/new-example.md
   ```

2. Add frontmatter at the top:
   ```markdown
   ---
   sidebar_position: 5
   ---

   # Your Title

   Content here...
   ```

3. Add to `sidebars.js` if not auto-detected:
   ```javascript
   {
     type: 'category',
     label: 'Basic Examples',
     items: [
       // existing items...
       'tutorials/basic/new-example',
     ],
   }
   ```

### New Guide

1. Create file in `docs/guides/`:
   ```bash
   touch docs/guides/new-guide.md
   ```

2. Add frontmatter with `sidebar_position`

3. Update `sidebars.js` if needed

## Configuration

### Site Settings

Edit `docusaurus.config.js`:

```javascript
const config = {
  title: 'DeepSpeed Course',
  tagline: 'Master distributed deep learning',
  url: 'https://yiqiao-yin.github.io',
  baseUrl: '/deepspeed-course/',
  // ...
};
```

### Navigation

Edit `sidebars.js` to change the documentation structure.

### Styling

Edit `src/css/custom.css` for custom styles.

## Deployment

### Automatic (GitHub Actions)

The site deploys automatically when changes are pushed to the `main` branch in the `docusaurus-docs/` directory.

Workflow file: `.github/workflows/deploy-docs.yml`

### Manual Deployment

```bash
# Build the site
npm run build

# Deploy to GitHub Pages
GIT_USER=<Your GitHub username> npm run deploy
```

## Search

The site uses [@easyops-cn/docusaurus-search-local](https://github.com/easyops-cn/docusaurus-search-local) for offline search:

- Press `Cmd+K` (Mac) or `Ctrl+K` (Windows/Linux) to open search
- No external API or account required
- Search index built at build time

## Customization

### Logo

Replace `static/img/logo.svg` with your own logo.

### Favicon

Replace `static/img/favicon.ico`.

### Colors

Edit CSS variables in `src/css/custom.css`:

```css
:root {
  --ifm-color-primary: #0066cc;
  /* ... */
}
```

## Troubleshooting

### Build Errors

```bash
# Clear cache and rebuild
npm run clear
npm run build
```

### Missing Dependencies

```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Local Development Issues

```bash
# Check Node version
node --version  # Should be >= 18.0

# Try fresh install
npm cache clean --force
npm install
```

## Contributing

1. Create a new branch
2. Make your changes
3. Test locally with `npm start`
4. Build with `npm run build`
5. Submit a pull request

## License

This documentation is part of the DeepSpeed Course project, released under the MIT License.
