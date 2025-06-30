```mermaid
flowchart TD
    A[Employee approaches scanner] --> B[System captures face image]
    B --> C[Process face image]
    C --> D{Face detected?}
    
    D -->|No| E[No Face Detected]
    E --> F[Return 'No Face' response]
    
    D -->|Yes| G{Face matches reference?}
    
    G -->|No Match| H[Access Denied]
    H --> I[Record access denial as 'Unknown']
    I --> J[Return 'Unknown' response]
    
    G -->|Match Found| K{Check employee status}
    
    K -->|Inactive| L[Access Denied]
    L --> M[Record access denial with employee info]
    M --> N[Return 'Denied' response]
    
    K -->|Active| O{Check confidence score}
    
    O -->|Low confidence \n< 70%| P[Access Denied]
    P --> Q[Record as 'Denied - Low confidence']
    Q --> R[Return 'Denied' response]
    
    O -->|Medium confidence \n70-80%| S{Check recent \nsuccessful matches}
    S -->|None in last \n10 minutes| T[Access Denied]
    T --> U[Record as 'Denied - Requires additional verification']
    U --> V[Return 'Denied' response]
    
    S -->|At least one \nrecent match| W[Access Granted]
    W --> X[Record as 'Granted - Multiple successful matches']
    X --> Y[Return 'Granted' response]
    
    O -->|High confidence \n> 80%| Z[Access Granted]
    Z --> AA[Record as 'Granted - High confidence match']
    AA --> AB[Return 'Granted' response]
```

## Face Recognition Access Control System Flowchart

This flowchart represents the actual implementation of the face recognition access control system. Note that the system determines access permission but does not physically control any door hardware.

The system:
1. Captures and processes face images
2. Matches faces against stored references
3. Checks employee status (active/inactive)
4. Evaluates match confidence levels
5. Records all access attempts
6. Returns appropriate responses

For high confidence matches with active employees, the system returns a "Granted" response, but the implementation of the physical access control (e.g., door unlocking) would require separate integration. 