// Main JavaScript file for the Facial Recognition Access Control system

// Handle logout
document.addEventListener('DOMContentLoaded', function() {
    // Check if logout link exists
    const logoutLink = document.querySelector('a[href="/logout"]');
    if (logoutLink) {
        logoutLink.addEventListener('click', function(e) {
            if (!confirm('Are you sure you want to logout?')) {
                e.preventDefault();
            }
        });
    }
    
    // Initialize any other UI elements
    initializeUI();
    
    // Initialize modals
    initializeModals();
    
    // Initialize profile dropdown
    initializeProfileDropdown();
});

// Initialize UI elements
function initializeUI() {
    // Add active class to current nav item
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav ul li a');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (currentPath === href) {
            link.classList.add('active');
        }
    });
}

// Initialize modal functionality
function initializeModals() {
    // Close modals when clicking outside
    window.onclick = function(event) {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        });
    };
    
    // Close buttons for modals
    const closeButtons = document.querySelectorAll('.modal .close');
    closeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const modal = this.closest('.modal');
            if (modal) {
                modal.style.display = 'none';
            }
        });
    });
}

// Initialize profile dropdown
function initializeProfileDropdown() {
    // Get DOM elements
    const profileIcon = document.getElementById('profileIcon');
    const profileDropdown = document.getElementById('profileDropdown');
    const changePasswordBtn = document.getElementById('changePasswordBtn');
    
    // Toggle dropdown when profile icon is clicked
    if (profileIcon && profileDropdown) {
        profileIcon.addEventListener('click', function(e) {
            e.stopPropagation();
            profileDropdown.classList.toggle('show');
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (profileDropdown.classList.contains('show') && 
                !profileDropdown.contains(e.target) && 
                e.target !== profileIcon) {
                profileDropdown.classList.remove('show');
            }
        });
    }
    
    // Change password functionality
    if (changePasswordBtn) {
        changePasswordBtn.addEventListener('click', function(e) {
            e.preventDefault();
            showChangePasswordModal();
        });
    }
    
    // Password form validation
    const changePasswordForm = document.getElementById('changePasswordForm');
    if (changePasswordForm) {
        changePasswordForm.addEventListener('submit', function(e) {
            const newPassword = document.getElementById('new_password').value;
            const confirmPassword = document.getElementById('confirm_password').value;
            
            if (newPassword !== confirmPassword) {
                e.preventDefault();
                alert('New password and confirmation do not match!');
            }
        });
    }
}

// Delete employee functions
function showDeleteModal(employeeId, employeeName) {
    const modal = document.getElementById('deleteModal');
    if (!modal) return;
    
    const deleteForm = document.getElementById('deleteForm');
    const confirmMessage = document.getElementById('deleteConfirmMessage');
    
    // Set the form action
    deleteForm.action = '/delete_employee/' + employeeId;
    
    // Update confirmation message
    confirmMessage.textContent = 'Are you sure you want to delete ' + employeeName + ' (ID: ' + employeeId + ')?';
    
    // Show the modal
    modal.style.display = 'block';
}

function hideDeleteModal() {
    const modal = document.getElementById('deleteModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Generic search function for tables
function searchTable() {
    const input = document.getElementById('searchInput');
    if (!input) return;
    
    const filter = input.value.toUpperCase();
    const table = document.querySelector('table');
    if (!table) return;
    
    const rows = table.getElementsByTagName('tr');
    
    // Loop through all table rows (skip the header row)
    for (let i = 1; i < rows.length; i++) {
        let row = rows[i];
        let cells = row.getElementsByTagName('td');
        let shouldDisplay = false;
        
        // Check if any cell contains the search query
        for (let j = 0; j < cells.length; j++) {
            let cell = cells[j];
            if (cell) {
                let textValue = cell.textContent || cell.innerText;
                if (textValue.toUpperCase().indexOf(filter) > -1) {
                    shouldDisplay = true;
                    break;
                }
            }
        }
        
        // Show or hide the row
        row.style.display = shouldDisplay ? '' : 'none';
    }
}

// Sort table by a specific column
function sortTable(columnIndex, direction = 'asc') {
    const table = document.querySelector('table');
    if (!table) return;
    
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    // Sort the rows
    rows.sort((a, b) => {
        const aValue = a.cells[columnIndex].textContent.trim();
        const bValue = b.cells[columnIndex].textContent.trim();
        
        // Try to parse as numbers if possible
        const aNum = parseFloat(aValue);
        const bNum = parseFloat(bValue);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return direction === 'asc' ? aNum - bNum : bNum - aNum;
        }
        
        // Fall back to string comparison
        return direction === 'asc' 
            ? aValue.localeCompare(bValue) 
            : bValue.localeCompare(aValue);
    });
    
    // Remove all rows from the table
    while (tbody.firstChild) {
        tbody.removeChild(tbody.firstChild);
    }
    
    // Add sorted rows back to the table
    rows.forEach(row => {
        tbody.appendChild(row);
    });
}

// Function to show change password modal
function showChangePasswordModal() {
    const modal = document.getElementById('changePasswordModal');
    if (modal) {
        modal.style.display = 'block';
    }
}

// Function to hide change password modal
function hideChangePasswordModal() {
    const modal = document.getElementById('changePasswordModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Create an init module to let other scripts call these functions
window.AppUI = {
    searchTable: searchTable,
    sortTable: sortTable,
    showDeleteModal: showDeleteModal,
    hideDeleteModal: hideDeleteModal,
    showChangePasswordModal: showChangePasswordModal,
    hideChangePasswordModal: hideChangePasswordModal
};
