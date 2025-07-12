let currentIndex = 0;
let images = [];

function openLightbox(imgElement) {
  const allImages = Array.from(document.querySelectorAll('.gallery img'));
  images = allImages.filter(img => img.parentElement.style.display !== 'none');
  currentIndex = images.indexOf(imgElement);

  document.getElementById("lightbox").style.display = "block";
  document.getElementById("lightbox-img").src = imgElement.src;
}

function closeLightbox() {
  document.getElementById("lightbox").style.display = "none";
}

function nextImage() {
  if (images.length === 0) return;
  currentIndex = (currentIndex + 1) % images.length;
  document.getElementById("lightbox-img").src = images[currentIndex].src;
}

function prevImage() {
  if (images.length === 0) return;
  currentIndex = (currentIndex - 1 + images.length) % images.length;
  document.getElementById("lightbox-img").src = images[currentIndex].src;
}

function filterImages(category) {
  const cards = document.querySelectorAll(".image-card");
  cards.forEach(card => {
    if (category === "All" || card.classList.contains(category)) {
      card.style.display = "block";
    } else {
      card.style.display = "none";
    }
  });
}
