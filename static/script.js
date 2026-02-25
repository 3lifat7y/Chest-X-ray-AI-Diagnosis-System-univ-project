/**
 * Main JavaScript file for handling X-ray image uploads, theme toggle, and doctor list display.
 * Integrates with a Flask API for multi-class classification (disease prediction).
 */
// DOM Elements
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('preview');
const themeLabel = document.getElementById('theme-label');
const themeToggle = document.querySelector('.theme-toggle');
const governorateSelect = document.getElementById('governorateSelect');
const removeButton = document.getElementById('removeButton');
const selectButton = document.querySelector('.select-button');
const analyzeButton = document.getElementById('analyzeButton');
const processingMessage = document.getElementById('processingMessage');
const processingText = processingMessage.querySelector('.processing-text');
const resultMessage = document.getElementById('resultMessage');
const resultText = resultMessage.querySelector('.result-text');

// Theme Toggle Functionality
themeToggle.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  toggleTheme();
});

function toggleTheme() {
  // Toggle between light and dark themes
  const isLight = document.body.classList.contains('light');
  document.body.classList.toggle('light');
  themeLabel.textContent = isLight ? '🌙' : '☀️';
  localStorage.setItem('theme', isLight ? 'dark' : 'light');

  // Update background overlay based on theme
  document.body.style.backgroundColor = isLight ? 'rgba(15, 17, 23, 0.3)' : 'rgba(244, 244, 249, 0.3)';
}

// File Upload Functionality (via button)
selectButton.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  fileInput.click();
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  handleFile(file);
  updateUploadText(file);
});

// Remove Image Functionality
removeButton.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  resetImagePreview();
});

// Drag-and-Drop Functionality
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false);
  document.body.addEventListener(eventName, preventDefaults, false);
});

['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false);
});

dropArea.addEventListener('drop', handleDrop, false);

function preventDefaults(e) {
  // Prevent default browser behavior for drag-and-drop
  e.preventDefault();
  e.stopPropagation();
}

function highlight() {
  // Highlight drop area during drag
  dropArea.classList.add('dragover');
}

function unhighlight() {
  // Remove highlight from drop area
  dropArea.classList.remove('dragover');
}

function handleDrop(e) {
  // Handle dropped file
  const dt = e.dataTransfer;
  const file = dt.files[0];
  handleFile(file);
  updateUploadText(file);
}

function handleFile(file) {
  // Process uploaded file and display preview
  if (!file) {
    alert('No file selected!');
    return;
  }

  if (!file.type.startsWith('image/')) {
    alert('Only image files are allowed!');
    resetImagePreview();
    return;
  }

  const reader = new FileReader();
  reader.onload = e => {
    const imageUrl = e.target.result;
    previewImage.src = imageUrl;
    previewImage.style.display = 'block';

    // Update background overlay for readability
    document.body.style.position = 'relative';
    document.body.style.backgroundBlendMode = 'overlay';
    document.body.style.backgroundColor = document.body.classList.contains('light')
      ? 'rgba(244, 244, 249, 0.3)'
      : 'rgba(15, 17, 23, 0.3)';

    const uploadContainer = document.getElementById('drop-area');
    uploadContainer.style.backgroundColor = 'var(--container)';
    uploadContainer.style.backgroundImage = 'none';
    uploadContainer.style.backgroundBlendMode = 'normal';
  };
  reader.onerror = () => {
    alert('Failed to read the image file!');
    resetImagePreview();
  };
  reader.readAsDataURL(file);
}

function updateUploadText(file) {
  // Update upload text based on file selection
  const uploadText = document.querySelector('.upload-text');
  if (file) {
    uploadText.textContent = `Selected: ${file.name}`;
    removeButton.style.display = 'block';
  } else {
    uploadText.textContent = 'Upload the X-ray Image 🩺';
    removeButton.style.display = 'none';
  }
}

function resetImagePreview() {
  // Reset image preview and related UI elements
  previewImage.src = '';
  previewImage.style.display = 'none';
  fileInput.value = '';
  updateUploadText(null);
  resultMessage.style.display = 'none';
  processingMessage.style.display = 'none';
  resultText.innerHTML = '';
}

// Doctors Data Structure (Complete)
const doctorsData = {
  alexandria: [
    { name: "Dr. Mohamed Ahmed Khames", specialty: "Chest Specialist", info: "Sidy Bishr: Cairo Street" },
    { name: "Dr. Rasha Daabis", specialty: "Chest Specialist", info: "Smouha: Fawzy Moaaz" },
    { name: "Dr. Haytham Emam", specialty: "Chest Specialist", info: "Sidy Bishr: Gamal Abdelnaser" },
    { name: "Dr. Hani Shaarawy", specialty: "Chest Specialist", info: "251 Port Said Street - Cleopatra" },
    { name: "Dr. Mahmoud Khalil", specialty: "Chest Specialist", info: "Moharram Bek" }
  ],
  cairo: [
    { name: "Dr. Shimaa Abou Eldahab", specialty: "Chest Specialist", info: "El Nasr Road, Nasr City" },
    { name: "Dr. Einas Badr", specialty: "Chest Specialist", info: "El Nasr Road, Nasr City" },
    { name: "Dr. Ashraf Adel Gomaa", specialty: "Chest Specialist", info: "El Manyal Street, Al Manyal" },
    { name: "Dr. Hisham Mohamed Shaheen", specialty: "Chest Specialist", info: "Tahrir Square, Downtown" },
    { name: "Dr. Rafek Adly", specialty: "Chest Specialist", info: "Ahmed Said Street, Al Abbasseya" }
  ],
  "kafr-el-sheikh": [
    { name: "Dr. Ahmed Hamdy Abo Elataa", specialty: "Chest Specialist", info: "Salah Salim Street, Kafr El-Sheikh City" },
    { name: "Dr. Ahmed Khalifa", specialty: "Chest Specialist", info: "Stadium Street, Kafr El-Sheikh City" },
    { name: "Dr. Ibrahim Al Munisi", specialty: "Chest Specialist", info: "Gamal Abdel Nasser Street, Qalin" },
    { name: "Dr. Ahmed Issa Elganiny", specialty: "Chest Specialist", info: "El Gesh Street, Kafr El-Sheikh City" },
    { name: "Dr. El Sayed Mohamed Abdel Salam", specialty: "Chest Specialist", info: "El Mohandeseen Street, El Riyad" }
  ],
  "el-qalyubia": [
    { name: "Dr. ElHusseiny ElSayed ElNemr", specialty: "Chest Specialist", info: "Orabi Street, Shoubra El-Kheima" },
    { name: "Dr. Abd Elghany Kasem", specialty: "Chest Specialist", info: "15th of May, Shoubra El-Kheima" },
    { name: "Dr. Ibrahim Farag Khater", specialty: "Chest Specialist", info: "15 May Street, Shoubra El-Kheima" },
    { name: "Dr. Ahmed Ragab", specialty: "Chest Specialist", info: "Port Said Street, Shoubra El-Kheima" },
    { name: "Dr. Ahmed Abu El-Fatth Tantawy", specialty: "Chest Specialist", info: "Qanater El-Kheiria" }
  ],
  "el-menufia": [
    { name: "Dr. Rabab Elwahsh", specialty: "Chest Specialist", info: "Shibin El-Kom" },
    { name: "Dr. Mohamed Kamel El Mehy", specialty: "Chest Specialist", info: "Shibin El-Kom" },
    { name: "Dr. Mohamed Ahmed Khalil", specialty: "Chest Specialist", info: "Shibin El-Kom" },
    { name: "Dr. Elessawy Soheil", specialty: "Chest Specialist", info: "Shibin El-Kom" },
    { name: "Dr. Moaaz Bakr", specialty: "Chest Specialist", info: "Birket El Sabea" }
  ],
  "el-daqahlia": [
    { name: "Dr. Mamdouh Ahmad Zeweeta", specialty: "Chest Specialist", info: "Tomehy Square, El-Mansoura" },
    { name: "Dr. Ahmed Kabil", specialty: "Chest Specialist", info: "Gehan Street, El-Mansoura" },
    { name: "Dr. Mohamed AbdElMonem", specialty: "Chest Specialist", info: "Khatab Square, Belkas" },
    { name: "Dr. Nermeen Ibrahim", specialty: "Chest Specialist", info: "El Nakhla Street, El-Mansoura" },
    { name: "Dr. Mohamed Hamoda", specialty: "Chest Specialist", info: "Ard ElMahlag, El Senbellawein" }
  ],
  sharqia: [
    { name: "Dr. Mohamed Sobhy Elgammal", specialty: "Chest Specialist", info: "Omar Afandy Street, El-Zagazig" },
    { name: "Dr. Hanan Mohamed Al Shahat", specialty: "Chest Specialist", info: "Abd Aziz Abaza Street, El-Zagazig" },
    { name: "Dr. Roshdy Yousef", specialty: "Chest Specialist", info: "Mekawy Tower, El-Zagazig" },
    { name: "Dr. Mohamed Al Shabrawy", specialty: "Chest Specialist", info: "Al Sagha Street, El-Zagazig" },
    { name: "Dr. Reda Elghamry", specialty: "Chest Specialist", info: "Tolba Ewida Street, El-Zagazig" }
  ],
  giza: [
    { name: "Dr. Waleed Ramadan", specialty: "Chest Specialist", info: "Al Haram Street, El-Haram" },
    { name: "Dr. Mohamed Aly", specialty: "Chest Specialist", info: "El Batal Ahmed Abdel Aziz Street, Dokki" },
    { name: "Dr. Hossam Mahmoud Hassanein", specialty: "Chest Specialist", info: "50 Mohi El Din Abu El Ezz Street" },
    { name: "Dr. Hossam Hosny Masoud", specialty: "Chest Specialist", info: "90 Ahmed Orabi Street" },
    { name: "Dr. Khaled Hussein", specialty: "Chest Specialist", info: "19 Al-Batal Ahmed Abdel Aziz Street" },
    { name: "Dr. Ahmed Sami", specialty: "Chest Specialist", info: "Dokki, Nile Street" },
    { name: "Dr. Mona Hassan", specialty: "Chest Specialist", info: "6th of October, Hosary Square" }
  ],
  beheira: [
    { name: "Dr. Mohamed Yassin Abdel Samea", specialty: "Chest Specialist", info: "Ahmed Orabi St., Damanhour" },
    { name: "Dr. Mohamed Abozeid", specialty: "Chest Specialist", info: "Elmahkma St., Kafr El Dawar" },
    { name: "Dr. Islam Wanas", specialty: "Chest Specialist", info: "Behind the General Hospital, Itay Al Barud" },
    { name: "Dr. Abdallah Abu Rehab", specialty: "Chest Specialist", info: "Doctors Tower, Orabi Street, Damanhour" },
    { name: "Dr. Mohamed Lotfy Khater", specialty: "Chest Specialist", info: "2 El Shafaey St, Kafr El Dawar" }
  ],
  gharbia: [
    { name: "Dr. Mohamed Abdelwahab", specialty: "Chest Specialist", info: "Elmoderya St, Tanta" },
    { name: "Dr. Ashraf Hamed", specialty: "Chest Specialist", info: "Shon Square, Mahalla" },
    { name: "Dr. Dalia Elsharawy", specialty: "Chest Specialist", info: "Shon Square, Mahalla" },
    { name: "Dr. Ragia Sharshar", specialty: "Chest Specialist", info: "Shoun Square, Mahalla" },
    { name: "Dr. Mina Botrous", specialty: "Chest Specialist", info: "El Bahr with El Khan, Tanta" }
  ],
  damietta: [
    { name: "Dr. Mohamed Mounir Mahmoud El Ros", specialty: "Chest Specialist", info: "Nile Corniche Street, Damietta" },
    { name: "Chest Disease Hospital", specialty: "Hospital", info: "Bab El Haras Square, Damietta" }
  ],
  "port-said": [
    { name: "Dr. Mohammed Sayed Abdelhafez", specialty: "Chest Specialist", info: "Port Said Chest Hospital" },
    { name: "Dr. Mohamed El Wazeir", specialty: "Chest Specialist", info: "Port Said City" }
  ],
  ismailia: [
    { name: "Dr. Muhammad Khattab", specialty: "Chest Specialist", info: "El Sheikh Zayed, Arab Al Hanadi Street" },
    { name: "Dr. Gamal Hamed Mohammed Al Beheiry", specialty: "Chest Specialist", info: "Al Bahri and Al Thalaini Street" },
    { name: "Dr. Osama Ishaq Michael", specialty: "Chest Specialist", info: "146 Saad Zaghloul St" }
  ],
  suez: [
    { name: "Dr. Ahmed Hussien Shady", specialty: "Chest Specialist", info: "8 Sports City Street-el Mallaha" }
  ],
  fayoum: [
    { name: "Dr. Mona Abd Almetal Helal", specialty: "Chest Specialist", info: "Bahr Tanhala St" },
    { name: "Dr. Moustafa Kamel Mohamed Alziady", specialty: "Chest Specialist", info: "Batal El Salam St" },
    { name: "Dr. Assem Fouad Mohamed El Essawy", specialty: "Chest Specialist", info: "Fayoum" },
    { name: "Dr. Afnan Abdel Halim", specialty: "Chest Specialist", info: "Fayoum" },
    { name: "Dr. Sherif Refaat Abdulfattah Alsayed", specialty: "Chest Specialist", info: "Fayoum" }
  ],
  "beni-suef": [
    { name: "Dr. Osama Ahmed", specialty: "Chest Specialist", info: "Beni Suef" },
    { name: "Dr. Aya Muhammad Abdel Salam", specialty: "Chest Specialist", info: "Beni Suef" },
    { name: "General Hospital", specialty: "Hospital", info: "near the city center" },
    { name: "Beni Suef University Hospital Chest", specialty: "Hospital", info: "Salah Salem St" }
  ],
  minya: [
    { name: "Dr. Hala Abdelhameed Mohamed", specialty: "Chest Specialist", info: "El Minya City" },
    { name: "Dr. Maged Moric", specialty: "Chest Specialist", info: "El Minya City" },
    { name: "Dr. Tarek Elmasry", specialty: "Chest Specialist", info: "El Minya City" },
    { name: "Dr. Esraa Abdel Kareem", specialty: "Chest Specialist", info: "Al-Minya Chest Hospital" }
  ],
  assiut: [
    { name: "Dr. Shady El Sawaaf", specialty: "Chest Specialist", info: "El Togareen, above El Torky" },
    { name: "Dr. Mohamed Fawzy Adam", specialty: "Chest Specialist", info: "El Hoda Mall Tower, 6th floor" },
    { name: "Dr. Mohammed Fawzy Barakat", specialty: "Chest Specialist", info: "Banks Square, Awkaf Tower No. 2" },
    { name: "Dr. Mohammad Gamal Abdalrahman", specialty: "Chest Specialist", info: "Youssry Ragheb St., Al Zohor Tower" },
    { name: "Dr. Wageeh Hassan", specialty: "Chest Specialist", info: "El Azhar St,Asmaa Allah El Hossna Square" }
  ],
  sohag: [
    { name: "Dr. Ahmed Fawzy", specialty: "Chest Specialist", info: "above Abu Dhabi Islamic Bank" },
    { name: "Dr. Hisham Gamal Ismail", specialty: "Chest Specialist", info: "Akhmim Road, Al-Zohour Tower" },
    { name: "Dr. Farouk Eid", specialty: "Chest Specialist", info: "Sohag Chest Hospital" },
    { name: "Dr. Maha Youssef", specialty: "Chest Specialist", info: "Sohag City" },
    { name: "Dr. Malek Abou Dahab", specialty: "Chest Specialist", info: "Sohag City" }
  ],
  qena: [
    { name: "Dr. Ahmed Nagy Fouad", specialty: "Chest Specialist", info: "Dandara Bridge Street" },
    { name: "Dr. Hagagy Mansour", specialty: "Chest Specialist", info: "Qena General Hospital Street" },
    { name: "Dr. Bassma Fayek", specialty: "Chest Specialist", info: "Qena Chest Hospital" }
  ],
  luxor: [
    { name: "Dr. Mohamed Hussein Abdallah", specialty: "Chest Specialist", info: "Luxor City" },
    { name: "Nile Medical Center", specialty: "Medical Center", info: "2 El Rawda El Shareife St" },
    { name: "Dr. Abo Elnaga Abdelraheem", specialty: "Chest Specialist", info: "Television St., Luxor City" },
    { name: "Luxor Medical Center", specialty: "Medical Center", info: "Luxor City" }
  ],
  aswan: [
    { name: "Dr. Mohamed Ahmed", specialty: "Chest Specialist", info: "Aswan University Hospital" },
    { name: "Dr. Ahmed Hassan", specialty: "Chest Specialist", info: "El-Kornish St., Aswan" },
    { name: "Dr. Khaled Mahmoud", specialty: "Chest Specialist", info: "El-Sail, Aswan" },
    { name: "Dr. Amr Abdelhamed", specialty: "Chest Specialist", info: "Aswan City" },
    { name: "Dr. Hassan Ibrahim", specialty: "Chest Specialist", info: "Kornish Al Nile, Aswan" }
  ],
  "red-sea": [
    { name: "Dr. Ramadan", specialty: "Chest Specialist", info: "Marsa Alam" },
    { name: "Dr. Khatab", specialty: "Chest Specialist", info: "Marsa Alam" },
    { name: "Chest Specialist at Red Sea Hospital", specialty: "Hospital", info: "Hurghada" },
    { name: "Chest Specialist at Royal Hospital", specialty: "Hospital", info: "1 Airport Road, Hurghada" },
    { name: "Chest Specialist at Nile Hospital", specialty: "Hospital", info: "El-Nasr, Airport Road, Hurghada" }
  ],
  "new-valley": [
    { name: "Dr. Ahmed Mahmoud Abdel Rahman", specialty: "Chest Specialist", info: "El Kharga General Hospital, El Kharga City" },
    { name: "Dr. Mohamed Hassan El Sayed", specialty: "Chest Specialist", info: "Dakhla Central Hospital, Mut Road" },
    { name: "Dr. Fatima Ali Ibrahim", specialty: "Chest Specialist", info: "Chest Clinic, Nasser Street, El Kharga" },
    { name: "Dr. Khaled Omar Mostafa", specialty: "Chest Specialist", info: "Al Farafra Medical Center, Farafra Oasis" },
    { name: "Dr. Samia Youssef Gad", specialty: "Chest Specialist", info: "El Kharga Polyclinic, Tahrir Square" }
  ],
  matrouh: [
    { name: "Dr. Ahmed Mostafa", specialty: "Chest Specialist", info: "Chest Hospital, Marsa Matrouh, Alexandria St" },
    { name: "Dr. Fatima El-Sayed", specialty: "Chest Specialist", info: "Matrouh General Hospital, Central Area" },
    { name: "Dr. Khaled Hassan", specialty: "Chest Specialist", info: "El Safa Clinics, Markaz Al Alamein" },
    { name: "Dr. Mona Abdel Rahman", specialty: "Chest Specialist", info: "Andalusia Hospitals North Coast, El Alamein" },
    { name: "Dr. Omar Ibrahim", specialty: "Chest Specialist", info: "Ras El-Hekma Emergency and Trauma Hosp" }
  ],
  "north-sinai": [
    { name: "Dr. Ahmed Mostafa", specialty: "Chest Specialist", info: "El Arish General Hospital, Al Masaeed Street" },
    { name: "Dr. Mohamed Abdel Rahman", specialty: "Chest Specialist", info: "Al Salam Medical Center, Coastal Road, El Arish" },
    { name: "Dr. Hassan Ibrahim", specialty: "Chest Specialist", info: "North Sinai Specialized Hospital, Rafah Road" },
    { name: "Dr. Khaled Sayed", specialty: "Chest Specialist", info: "El Masaeed Beach Area, El Arish" },
    { name: "Dr. Fatima Ali", specialty: "Chest Specialist", info: "Bir al-Abed Medical Center, Bir al-Abed" }
  ],
  "south-sinai": [
    { name: "South Sinai Hospital- Sharm El Sheikh", specialty: "Hospital", info: "10 Ras Kennedy El-Salam, Qesm Sharm Ash Sheikh" },
    { name: "Sharm International Hospital", specialty: "Hospital", info: "Hay El Nour, Sharm El Sheikh" },
    { name: "El Tor General Hospital", specialty: "Hospital", info: "El Tor, South Sinai" },
    { name: "Dahab Specialized Hospital", specialty: "Hospital", info: "Mashraba, Dahab" },
    { name: "Taba Medical Center", specialty: "Medical Center", info: "Taba, near the Taba border crossing" }
  ],
  mansoura: [
    { name: "Dr. Youssef Mahmoud", specialty: "Chest Specialist", info: "Mansoura Downtown" }
  ]
};

function displayDoctors(governorate) {
  const doctorsContainer = document.getElementById('doctorsContainer');
  const doctorsList = document.getElementById('doctorsList');

  doctorsList.innerHTML = '';

  const doctors = doctorsData[governorate] || [];

  if (doctors.length > 0) {
    doctors.forEach(doctor => {
      const doctorCard = document.createElement('div');
      doctorCard.className = 'doctor-card';
      doctorCard.innerHTML = `
        <div class="doctor-name">${doctor.name}</div>
        <div class="doctor-specialty">${doctor.specialty}</div>
        <div class="doctor-info">${doctor.info}</div>
      `;
      doctorsList.appendChild(doctorCard);
    });
    doctorsContainer.style.display = 'block';
  } else {
    doctorsList.innerHTML = '<div class="doctor-card">Please Select a Governorate</div>';
    doctorsContainer.style.display = 'block';
  }
}

governorateSelect.addEventListener('change', (e) => {
  // Update doctor list when governorate changes
  const selectedGovernorate = e.target.value;
  displayDoctors(selectedGovernorate);
  e.stopPropagation();
});

// Analyze Image Functionality
analyzeButton.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  processingMessage.style.display = 'flex';
  processingText.textContent = "Processing...";
  processingText.style.color = 'var(--text)';
  resultMessage.style.display = 'none';
  resultText.innerHTML = '';
  analyzeImage();
});

function analyzeImage() {
  // Send image to API for analysis
  const file = fileInput.files[0];
  if (!file) {
    processingMessage.style.display = 'none';
    resultMessage.style.display = 'flex';
    resultText.innerHTML = '<div style="color: #ff4444;">Please upload an image first!</div>';
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  console.log("Sending request to API with file:", file.name);

  fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    console.log("Response status:", response.status);
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    console.log("API response:", data);
    processingMessage.style.display = 'none';
    resultMessage.style.display = 'flex';

    if (data.error) {
      resultText.innerHTML = `<div style="color: #ff4444;">Error: ${data.error}</div>`;
    } else {
      const diseases = data.diseases;
      const numDiseases = data.num_diseases;

      if (numDiseases === 0 && diseases[0].name === 'No Finding') {
        resultText.innerHTML = `
          <div style="color: #4CAF50;">The X-ray appears normal.</div>
        `;
      } else {
        let resultHTML = diseases.map(disease => `
          <div>
            <span style="color:#FFFFF0;">Detected Disease: </span>
            <span style="color:#e13a31;">${disease.name}</span>
          </div>
          <div style="margin: 10px 0;">
            <span style="color:#FFFFF0;">Description: </span>
            <span style="color:#e13a31;">${disease.description}</span>
          </div>
        `).join('');
        resultHTML += '<div style="color: #4CAF50;">Please consult a doctor. Select your governorate below to find a specialist.</div>';
        resultText.innerHTML = resultHTML;
      }
    }
  })
  .catch(error => {
    console.error("Error during fetch:", error);
    processingMessage.style.display = 'none';
    resultMessage.style.display = 'flex';
    resultText.innerHTML = `<div style="color: #ff4444;">Error: Failed to connect to the server. Ensure the API is running on http://localhost:5000 and try again.</div>`;
  });
}

// Initialize on Page Load
document.addEventListener('DOMContentLoaded', () => {
  // Apply saved theme and initialize UI
  const theme = localStorage.getItem('theme');
  if (theme === 'light') {
    document.body.classList.add('light');
    themeLabel.textContent = '☀️';
    document.body.style.backgroundColor = 'rgba(244, 244, 249, 0.3)';
  } else {
    document.body.classList.remove('light');
    themeLabel.textContent = '🌙';
    document.body.style.backgroundColor = 'rgba(15, 17, 23, 0.3)';
  }
  removeButton.style.display = 'none';
});