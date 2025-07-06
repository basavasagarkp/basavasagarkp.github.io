## Local Development Setup

### Prerequisites
- Ruby (this project uses Ruby 2.6+)
- Bundler gem manager

### Installation Steps

1. **Install Bundler** (if not already installed):
   ```bash
   gem install --user-install bundler -v 2.4.22
   ```

2. **Add gem bin directory to PATH**:
   ```bash
   export PATH="$PATH:$HOME/.gem/ruby/2.6.0/bin"
   ```

3. **Install dependencies**:
   ```bash
   bundle install --path vendor/bundle
   ```

4. **Start the development server**:
   ```bash
   bundle exec jekyll serve
   ```

5. **View the website**:
   Open your browser and go to `http://localhost:4000`

### Troubleshooting

- If you encounter permission errors, use `--user-install` flag with gem commands
- If using system Ruby, the `--path vendor/bundle` flag installs gems locally to avoid permission issues
- Press `Ctrl+C` to stop the development server
