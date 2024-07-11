document.addEventListener('DOMContentLoaded', (event) => {
    document.querySelectorAll('pre').forEach((block) => {
      // Create button
      const button = document.createElement('button');
      button.innerText = 'Copy';
      button.className = 'copy-button';
      
      // Add button to pre block
      block.style.position = 'relative';
      block.appendChild(button);
      
      // Add click event
      button.addEventListener('click', () => {
        const code = block.querySelector('code') || block;
        navigator.clipboard.writeText(code.innerText).then(() => {
          // Visual feedback
          button.innerText = 'Copied!';
          setTimeout(() => {
            button.innerText = 'Copy';
          }, 2000);
        }, (err) => {
          console.error('Failed to copy: ', err);
          button.innerText = 'Failed';
        });
      });
    });
  });