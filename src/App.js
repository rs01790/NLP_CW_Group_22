import React, { useState } from 'react';
import { ChakraProvider, Heading, FormControl, FormLabel, Input, Button, Box, AlertDialog, AlertDialogOverlay, AlertDialogContent, AlertDialogHeader, AlertDialogBody, AlertDialogFooter, AlertDialogCloseButton } from '@chakra-ui/react';

function App() {
    const [sentence, setSentence] = useState('');
    const [emotion, setEmotion] = useState('');
    const [isPopupOpen, setIsPopupOpen] = useState(false);

    const handleAnalyze = async () => {
        if (sentence.trim() === '') {
            setIsPopupOpen(true); // Show the popup if no text is entered
            return;
        }

        try {
            const response = await fetch('http://localhost:3005/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentence }),
            });

            if (response.ok) {
                const { modifiedData } = await response.json();
                setEmotion(modifiedData);
                setSentence('');
            } else {
                console.error('Error:', response.statusText);
            }
        } catch (error) {
            console.error('Error:', error);
        }
    };

    const handleClosePopup = () => {
        setIsPopupOpen(false);
    };

    return (
        <ChakraProvider>
            <Heading as="h1" mb="4">
                Emotion Analyzer
            </Heading>
            <FormControl id="sentence" mb="4">
                <FormLabel>Enter a sentence to analyze its emotion</FormLabel>
                <Input value={sentence} onChange={e => setSentence(e.target.value)} />
            </FormControl>
            <Button colorScheme="blue" onClick={handleAnalyze}>
                Analyze
            </Button>
            {emotion && (
                <Box mt="4">
                    <Heading as="h2">Predicted Emotion:</Heading>
                    <Box p="4" borderWidth="1px" borderRadius="md">
                        {emotion}
                    </Box>
                </Box>
            )}
            <AlertDialog isOpen={isPopupOpen} onClose={handleClosePopup}>
                <AlertDialogOverlay />
                <AlertDialogContent>
                    <AlertDialogHeader>Please enter a sentence</AlertDialogHeader>
                    <AlertDialogCloseButton />
                    <AlertDialogBody>
                        The sentence field cannot be empty. Please enter a sentence to analyze its emotion.
                    </AlertDialogBody>
                    <AlertDialogFooter>
                        <Button colorScheme="blue" onClick={handleClosePopup} ml={3}>
                            OK
                        </Button>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>
        </ChakraProvider>
    );
}

export default App;
