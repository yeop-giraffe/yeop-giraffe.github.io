# yeop-giraffe.github.io

Personal website for Seungyeop Lee, built with the [al-folio](https://github.com/alshedivat/al-folio) Jekyll theme.

## Local Development

Install Ruby dependencies:

```sh
bundle install
```

Run the site locally:

```sh
bundle exec jekyll serve
```

Then open `http://localhost:4000`.

## Deployment

Push this repository to GitHub as `yeop-giraffe.github.io`. The included GitHub Actions workflow builds the Jekyll site and deploys it to GitHub Pages.

The site URL is configured in `_config.yml` as:

```yml
url: https://yeop-giraffe.github.io
baseurl:
```
