import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  TextField,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  Paper,
  Container,
  Typography,
  Chip,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
} from "@mui/material";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";
import ArrowDownwardIcon from "@mui/icons-material/ArrowDownward";

const getChipColor = (subRequest) => {
  if (!subRequest) return "default";

  const colors = [
    "primary",
    "secondary",
    "warning",
    "success",
    "info",
    "error",
  ];

  // Create a hash-based index to pick a color consistently
  const hash = subRequest
    .split("")
    .reduce((acc, char) => acc + char.charCodeAt(0), 0);

  return colors[hash % colors.length]; // Pick color based on hash
};

const EmailClassificationUI = () => {
  // Initial request data from the sample provided in the PDF
  const [requests, setRequests] = useState([]);
  const [search, setSearch] = useState("");
  const [open, setOpen] = useState(false);
  const [newRequestType, setNewRequestType] = useState("");
  // Accept comma-separated values for sub request types
  const [newSubRequestType, setNewSubRequestType] = useState("");
  const [error, setError] = useState("");
  const [sortOrder, setSortOrder] = useState("asc");
  const [successMessage, setSuccessMessage] = useState("");

  // Filter based on request type (search ignores sub request types for simplicity)


  useEffect(() => {
    const fetchCategories = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:8000/categories");
        setRequests(response.data.categories);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    fetchCategories();
  }, []);

  const toggleSortOrder = () => {
    setSortOrder(sortOrder === "asc" ? "desc" : "asc");
  };

  const handleAddRequest = async () => {
    // Normalize input values
    const normalizedNewRequestType = newRequestType.trim().toLowerCase();
    const newSubRequests = newSubRequestType
      .split(",")
      .map((sub) => sub.trim().toLowerCase())
      .filter(Boolean);

    // Check if request type already exists
    const existingRequest = requests.find(
      (req) => req.requestType.trim().toLowerCase() === normalizedNewRequestType
    );

    if (existingRequest) {
      // If request type exists, check if the subrequest types also exist
      const existingSubRequests = existingRequest.subRequestType.map((sub) =>
        sub.trim().toLowerCase()
      );

      const allSubRequestsExist = newSubRequests.every((sub) =>
        existingSubRequests.includes(sub)
      );

      if (allSubRequestsExist) {
        setError("This request type with the same sub-request types already exists.");
        return;
      }
    }

    try {
      const response = await axios.post("http://127.0.0.1:8000/add-category", {
        id: requests.length + 1, // Assuming API assigns an ID
        requestType: newRequestType.trim(),
        subRequestType: newSubRequests.length > 0 ? newSubRequests : ["N/A"],
      });

      if (response?.data?.message) {  // ✅ Check if API confirms success
        // ✅ Update the table with new data without refetching
        setRequests((prevRequests) => [
          ...prevRequests,
          {
            id: response.data.id, // Assuming API returns an ID for the new request
            requestType: newRequestType.trim(),
            subRequestType: newSubRequests.length > 0 ? newSubRequests : ["N/A"],
          },
        ]);
        setSuccessMessage(response.data.message)
        setTimeout(() => setSuccessMessage(""), 2000);

        setOpen(false);
        setNewRequestType("");
        setNewSubRequestType("");
        setError("");
      } else {
        setError("Failed to add request.");
      }
    } catch (error) {
      setError(error.response?.data?.detail || "Error adding request.");
    }
  };


  const handleCancel = () => {
    setOpen(false);
    setError("");
    setNewRequestType("");
    setNewSubRequestType("");
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Email Request Classification
      </Typography>
      {successMessage && ( // ✅ Display success banner
        <Alert severity="success" sx={{ mb: 2 }}>
          {successMessage}
        </Alert>
      )}
      <TextField
        fullWidth
        label="Search Requests"
        variant="outlined"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        sx={{ mb: 3 }}
      />
      <Button
        variant="contained"
        onClick={() => {
          setOpen(true);
          setError("");
        }}
        sx={{ mb: 3 }}
      >
        Add New Request
      </Button>
      <Paper sx={{ overflow: "hidden" }}>
        <Table>
          <TableHead>
            <TableRow sx={{ backgroundColor: "primary.main" }}>
              <TableCell
                onClick={toggleSortOrder}
                sx={{
                  color: "white",
                  fontWeight: "bold",
                  borderRight: "1px solid white",
                  cursor: "pointer",
                }}
              >
                Request Type{" "}
                {sortOrder === "asc" ? (
                  <ArrowUpwardIcon fontSize="small" />
                ) : (
                  <ArrowDownwardIcon fontSize="small" />
                )}
              </TableCell>
              <TableCell sx={{ color: "white", fontWeight: "bold" }}>
                Sub Request Type
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {requests?.length > 0 && requests?.map((req) => (
              <TableRow key={req.id}>
                <TableCell sx={{ borderRight: "1px solid #ccc" }}>
                  {req.requestType}
                </TableCell>
                <TableCell>
                  {req.subRequestType.length > 0 ? (
                    req.subRequestType.map((sub, index) => (
                      <Chip
                        key={index}
                        label={sub}
                        color={getChipColor(sub)}
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))
                  ) : (
                    "N/A"
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      {/* Modal for adding a new request */}
      <Dialog open={open} onClose={handleCancel}>
        <DialogTitle>Add New Request</DialogTitle>
        <DialogContent>
          {error && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          <TextField
            autoFocus
            margin="dense"
            label="Request Type"
            fullWidth
            variant="outlined"
            value={newRequestType}
            onChange={(e) => setNewRequestType(e.target.value)}
          />
          <TextField
            margin="dense"
            label="Sub Request Types (comma separated)"
            fullWidth
            variant="outlined"
            value={newSubRequestType}
            onChange={(e) => setNewSubRequestType(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancel}>Cancel</Button>
          <Button onClick={handleAddRequest} variant="contained">
            Add
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default EmailClassificationUI;
